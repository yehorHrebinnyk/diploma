import torch

from tqdm import tqdm


def evaluate(dataloader,
         model,
         compute_loss,
         half_precision=True):
    device = next(model.parameters()).device
    half = device.type != "cpu" and half_precision
    if half:
        model.half()

    model.eval()

    mean_loss = torch.zeros(3, device=device)

    for bi, (img, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        targets = targets.to(device)

        with torch.no_grad():
            out, train_out = model(img)

            if compute_loss:
                loss = compute_loss([x.float() for x in train_out], targets)[1][:3]
                mean_loss = (mean_loss * bi + loss) / (bi + 1)
        break

    return mean_loss.mean()
