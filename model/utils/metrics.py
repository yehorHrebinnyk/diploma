import numpy as np
import torch

from .general import box_iou


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, num_classes, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
            Return intersection-over-union (Jaccard index) of boxes.
            Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
            Arguments:
                detections (Array[N, 6]), x1, y1, x2, y2, conf, class
                labels (Array[M, 5]), class, x1, y1, x2, y2
            Returns:
                None, updates confusion matrix accordingly
        """

        detections = detections[detections[:, 4] > self.conf]
        ground_truth_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        candidates = torch.where(iou > self.iou_thres)

        if candidates[0].shape[0]:
            matches = torch.cat((torch.stack(candidates, 1), iou[candidates[0], candidates[1]]), 1).cpu().numpy()
            if candidates[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        amount = matches.shape[0] > 0
        labels_i, pred_i, _ = matches.transpose().astype(np.int16)
        for i, ground_truth in enumerate(ground_truth_classes):
            j = labels_i == i

            if amount and sum(j) == 1:
                self.matrix[ground_truth, detections[pred_i[j]]] += 1  # correct
            else:
                self.matrix[self.num_classes, ground_truth] += 1  # # background FP

        if amount:
            for i, class_ in enumerate(detection_classes):
                if not any(pred_i == i):
                    self.matrix[class_, self.num_classes] += 1  # background FN

    def matrix(self):
        return self.matrix
