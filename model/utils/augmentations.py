import numpy as np
import cv2
import random
import math


def filter_boxes(box1, box2, wh_threshold=2, ar_threshold=100, area_threshold=0.1, eps=1e-16):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_threshold) & (h2 > wh_threshold) & (w2 * h2 / (w1 * h1 + eps) > area_threshold) & (
            aspect_ratio < ar_threshold)


def mixup(img, labels, img2, labels2):
    mixup_ratio = np.random.beta(32.0, 32.0)
    img = (img * mixup_ratio + img2 * (1 - mixup_ratio)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return img, labels


def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    if h_gain or s_gain or v_gain:
        random_gains = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        data_type = img.dtype

        x = np.arange(0, 256, dtype=random_gains.dtype)
        lut_hue = ((x * random_gains[0]) % 180).astype(data_type)
        lut_sat = np.clip(x * random_gains[1], 0, 255).astype(data_type)
        lut_val = np.clip(x * random_gains[2], 0, 255).astype(data_type)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


def random_augmentation(img, targets=(), img_size=640, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0):
    # targets[:, 1:] = xywhn2xyxy(targets[:, 1:], w=img_size * 2, h=img_siz * 2)

    center = np.eye(3)
    center[0, 2] = -img_size / 2
    center[1, 2] = -img_size / 2

    perspec = np.eye(3)
    perspec[2, 0] = random.uniform(-perspective, perspective)
    perspec[2, 1] = random.uniform(-perspective, perspective)

    rotation = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    scale_value = random.uniform(1 - scale, 1 + scale)
    rotation[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale_value)

    shear_matrix = np.eye(3)
    shear_matrix[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    shear_matrix[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    translation = np.eye(3)
    translation[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * img_size
    translation[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * img_size

    transform_matrix = translation @ shear_matrix @ rotation @ perspec @ center
    if (transform_matrix != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(img, transform_matrix, dsize=(img_size, img_size), borderValue=(120, 120, 120))
        else:
            img = cv2.warpAffine(img, transform_matrix[:2], dsize=(img_size, img_size), borderValue=(120, 120, 120))

    amount_of_targets = len(targets)
    if amount_of_targets:
        xy = np.ones((amount_of_targets * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(amount_of_targets * 4, 2)
        xy = xy @ transform_matrix.T
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(amount_of_targets, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, amount_of_targets).T

        new[:, [0, 2]] = new[:, [0, 2]].clip(0, 640)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, 640)
        i = filter_boxes(box1=targets[:, 1:5].T * scale_value, box2=new.T, area_threshold=0.01)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets
