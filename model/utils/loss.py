import torch
import torch.nn as nn

from .torch_utils import is_parallel, bbox_iou


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    def __init__(self, loss_fn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = self.loss_fn.reduction
        self.loss_fn.reduction = 'none'

    def forward(self, predicted, true):
        loss = self.loss_fn(predicted, true)
        predicted_probabilty = torch.sigmoid(loss)
        result = true * predicted_probabilty + (1 - true) * (1 - predicted_probabilty)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1 - result) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ComputeLoss:
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device
        hyperparameters = model.hyp

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyperparameters["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyperparameters["obj_pw"]], device=device))

        self.class_positive, self.class_negative = smooth_BCE(eps=hyperparameters.get("label_smoothing", 0.0))

        gamma = hyperparameters["fl_gamma"]
        if gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

        detect_layer = model.module.model[-1] if is_parallel(model) else model.model[-1]
        print('testug', next(detect_layer.parameters()).device)
        self.balance = {3: [4.0, 1.0, 0.4]}.get(detect_layer.number_layers, [4.0, 1.0, 0.25, 0.06, .02])
        self.stride_index_16 = list(detect_layer.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, hyperparameters, autobalance
        self.number_anchors = detect_layer.number_anchors
        self.number_classes = detect_layer.number_classes
        self.number_layers = detect_layer.number_layers
        self.anchors = detect_layer.anchors

    def build_targets(self, predicted, targets):
        number_anchors, number_targets = self.number_anchors, targets.shape[0]
        true_cls, true_box, indices, anchors_list = [], [], [], []
        grid_size = torch.ones(7, device=targets.device).long()
        anchor_indices_grid = torch.arange(number_anchors, device=targets.device).float() \
            .view(number_anchors, 1) \
            .repeat(1, number_targets)
        targets = torch.cat((targets.repeat(number_anchors, 1, 1), anchor_indices_grid[:, :, None]),
                                               2)
        bias = 0.5
        offset = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * bias

        for i in range(self.number_layers):
            anchors = self.anchors[i]
            grid_size[2:6] = torch.tensor(predicted[i].shape)[[3, 2, 3, 2]]
            grid_targets = targets * grid_size
            
            if number_targets:
                # erase true boxes which are too big or too small (relatively anchors)
                ratio_of_anchors_and_true_boxes = grid_targets[:, :, 4:6] / anchors[:, None]
                # find max_relation (whether it's too small or too big)
                normal_size_boxes = torch.max(
                    ratio_of_anchors_and_true_boxes,
                    1. / ratio_of_anchors_and_true_boxes
                ).max(2)[0] < self.hyp['anchor_t']

                # filter boxes
                grid_targets = grid_targets[normal_size_boxes]

                grid_xy = grid_targets[:, 2:4]
                grid_xy_inversed = grid_size[[2, 3]] - grid_xy

                is_small_tail_to_offset = torch.lt(grid_xy % 1., bias)
                is_cords_not_on_edge = torch.gt(grid_xy, 1.)

                # left offset for example 4.4 -> 4, 3.1 -> 3, 6.234 -> 6
                x_cords_to_offset, y_cords_to_offset = torch.logical_and(is_small_tail_to_offset,
                                                                         is_cords_not_on_edge).T

                # right offset for example, here we count how distance between close right cell and cell that we have
                # for example x = 24.6; map_size = 40, then x_reversed = 40 - 24.6 = 15.4; 15.4 -> 15;
                is_small_tail_to_offset_inversed = torch.lt(grid_xy_inversed % 1., bias)
                is_cords_not_on_edge_inversed = torch.gt(grid_xy_inversed, 1.)
                inversed_x_cords_to_offset, inversed_y_cords_to_offset = torch.logical_and(
                    is_small_tail_to_offset_inversed,
                    is_cords_not_on_edge_inversed).T

                # create targets with different offsets (without offset, right offset, left offset)
                boxes_indices_without_offset = torch.ones_like(x_cords_to_offset)
                indixes_targets_for_offset = torch.stack((
                    boxes_indices_without_offset,
                    x_cords_to_offset,
                    y_cords_to_offset,
                    inversed_x_cords_to_offset,
                    inversed_y_cords_to_offset
                ))

                grid_targets = grid_targets.repeat((5, 1, 1))[indixes_targets_for_offset]
                offsets_grid = (torch.zeros_like(grid_xy)[None] + offset[:, None])[indixes_targets_for_offset]
            else:
                grid_targets = targets[0]
                offsets_grid = 0

            batch, class_ = grid_targets[:, :2].long().T
            grid_xy = grid_targets[:, 2:4]
            grid_wh = grid_targets[:, 4:6]

            grid_indixes = (grid_xy - offsets_grid).long()
            grid_i, grid_j = grid_indixes.T

            # final step, making targets
            anchors_indixes = grid_targets[:, 6].long()
            indices.append((batch,
                            anchors_indixes,
                            grid_j.clamp_(0, grid_size[3] - 1),
                            grid_i.clamp_(0, grid_size[2] - 1)))

            position_in_cells = grid_xy - grid_indixes
            true_box.append(torch.cat((position_in_cells, grid_wh), 1))
            anchors_list.append(anchors[anchors_indixes])
            true_cls.append(class_)

        return true_cls, true_box, indices, anchors_list

    def __call__(self, predicted, targets):
        device = targets.device
        loss_cls, loss_box, loss_obj = torch.zeros(1, device=device), \
                                       torch.zeros(1, device=device), \
                                       torch.zeros(1, device=device)
        true_cls, true_box, indices, anchors = self.build_targets(predicted, targets)
        for i, layer_prediction in enumerate(predicted):
            batch_indexes, anchors_indexes, y, x = indices[i]
            targets_count = batch_indexes.shape[0]
            true_obj = torch.zeros_like(layer_prediction[..., 0], device=device)

            if targets_count:
                prediction_subset = layer_prediction[batch_indexes, anchors_indexes, y, x]

                predicted_xy = prediction_subset[:, :2].sigmoid() * 2 - 0.5
                predicted_wh = (prediction_subset[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                predicted_box = torch.cat((predicted_xy, predicted_wh), 1)
                iou = bbox_iou(predicted_box.T, true_box[i], x1y1x2y2=False, CIoU=True)
                loss_box += (1.0 - iou).mean()

                true_obj[batch_indexes, anchors_indexes, y, x] = (1.0 - self.gr) + self.gr * iou.detach()\
                    .clamp(0).type(true_obj.dtype)

                if self.number_classes > 1:
                    targets_classes = torch.full_like(prediction_subset[:, 5:], self.class_negative, device=device)
                    targets_classes[range(targets_count), true_cls[i]] = self.class_positive
                    loss_cls += self.BCEcls(prediction_subset[:, 5:], targets_classes)

            obj_layer = self.BCEobj(layer_prediction[..., 4], true_obj)
            loss_obj += obj_layer * self.balance[i]

            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obj_layer.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.stride_index_16] for x in self.balance]

        loss_box *= self.hyp["box"]
        loss_obj *= self.hyp["obj"]
        loss_cls *= self.hyp["cls"]
        bs = true_obj.shape[0]
        loss = loss_box + loss_obj + loss_cls
        return loss * bs, torch.cat((loss_box, loss_cls, loss_obj, loss))
