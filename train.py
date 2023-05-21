import argparse
import os
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np
import json
import time

from tqdm import tqdm
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp

from yolo import Yolo
from model.utils.datasets import create_dataloader
from model.utils.loss import ComputeLoss
from model.utils.general import one_cycle

from val import evaluate

def set_description(title, items):
    columns = title.split(" | ")
    desc_string = ""

    for column, item in zip(columns, items):
        desc_string += str(item) + ' ' * max(len(column) - len((str(item))), 0) + " | "

    return desc_string[:-2]


def train(hyp, args, device):
    save_dir = Path("./saves/")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    number_layers = 3
    workers = args.workers
    epochs = args.epochs
    batch_size = args.bs
    total_batch_size = args.total_batch_size
    rank = args.global_rank
    cuda = device.type != 'cpu'
    start_epoch = 0

    with open("./classes.txt", "r") as f:
        names = eval(f.read())

    num_classes = len(names)

    model = Yolo(number_classes=num_classes).to(device)
    nominal_batch_size = 64

    accumulate = max(round(nominal_batch_size / batch_size), 1)
    hyp['weight_decay'] *= total_batch_size * accumulate / nominal_batch_size

    param_group0, param_group1, param_group2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            param_group2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            param_group0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            param_group1.append(v.weight)

    optimizer = optim.Adam(param_group0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    optimizer.add_param_group(
        {'params': param_group1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': param_group2})  # add pg2 (biases)
    del param_group0, param_group1, param_group2

    lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # train, train_ds = create_dataloader(args.train,
    #                                     batch_size=args.bs,
    #                                     hyp=hyp,
    #                                     augment=True,
    #                                     rank=rank,
    #                                     workers=workers)

    # number_batches = len(train)

    # if rank in [-1, 0]:
    train, _ = create_dataloader(args.val,
                                 batch_size=args.bs,
                                 hyp=hyp,
                                 augment=False,
                                 rank=rank,
                                 workers=workers)

    number_batches = len(train)

    if cuda and rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    hyp['box'] *= 3. / number_layers  # scale to layers
    hyp['cls'] *= num_classes / 80. * 3. / number_layers  # scale to classes and layers
    hyp['obj'] *= 1 ** 2 * 3. / num_classes  # scale to image size and layers
    model.num_classes = num_classes
    model.hyp = hyp
    model.gr = 1.0
    model.names = names

    t0 = time.time()
    number_warmup_batches = max(round(hyp['warmup_epochs'] * number_batches), 1000)
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)

    title_string = "EPOCHS | MEM CUDA | LOSS BOX | LOSS CLS | LOSS OBJ | FINAL LOSS | TARGETS"
    print(title_string)

    for epoch in range(start_epoch, epochs):
        model.train()
        mean_loss = torch.zeros(4, device=device)

        pbar = enumerate(train)
        pbar = tqdm(pbar, total=number_batches)
        optimizer.zero_grad()

        for i, (imgs, targets) in pbar:
            # intergrated_batches = i + number_batches * epoch
            # imgs = imgs.to(device, non_blocking=True).float() / 255.0
            #
            # if intergrated_batches <= number_warmup_batches:
            #     xi = [0, number_warmup_batches]
            #     accumulate = max(1, np.interp(intergrated_batches, xi, [1, nominal_batch_size / batch_size]).round())
            #     for j, x in enumerate(optimizer.param_groups):
            #         x['lr'] = np.interp(intergrated_batches, xi,
            #                             [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            #         if 'momentum' in x:
            #             x['momentum'] = np.interp(intergrated_batches, xi,
            #                                       [hyp['warmup_momentum'], hyp['momentum']])
            #
            # with amp.autocast(enabled=cuda):
            #     pred = model(imgs)
            #     loss, loss_items = compute_loss(pred, targets.to(device))
            #
            # scaler.scale(loss).backward()
            #
            # if intergrated_batches - last_opt_step >= accumulate:
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     last_opt_step = intergrated_batches
            #
            # mean_loss = (mean_loss * i + loss_items) / (i + 1)
            # memory_used = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
            # epochs_left = f'{epoch}/{epochs - 1}'
            #
            # pbar.set_description(set_description(title_string, [epochs_left, memory_used, f'{mean_loss[0].item():.4f}',
            #                                                     f'{mean_loss[1].item():.4f}',
            #                                                     f'{mean_loss[2].item():.4f}',
            #                                                     f'{mean_loss[3].item():.4f}',
            #                                                     targets.shape[0]]))

            results = evaluate(train, model, compute_loss, half_precision=True)
            torch.save(model.state_dict(), save_dir / f"loss_{(mean_loss.mean().item()):.4f}.pt")
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="Path to train folder")
    parser.add_argument("--val", type=str, help="Path to validation folder")
    parser.add_argument("--hyp", type=str, help="Path to json file with hyperparams", default='./hyp.json')
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    args = parser.parse_args()

    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.total_batch_size = args.bs

    with open(args.hyp) as f:
        hyp = json.load(f)

    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        assert args.bs % args.world_size == 0, '--batch-size must be multiple of CUDA device count'
        args.bs = args.total_batch_size // args.world_size

    train(hyp, args, device)
