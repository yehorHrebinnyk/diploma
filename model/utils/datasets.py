import numpy as np
import random
import torch
import os

from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from .augmentations import random_augmentation, augment_hsv
from .general import xyxy2xywhn, xywhn2xyxy
from tqdm import tqdm


def create_dataloader(path, img_size=640, batch_size=16, hyp=None, augment=False,
                      cache=False, workers=8, rank=-1, world_size=1):
    dataset = LoadImagesAndLabels(path,
                                  img_size=img_size,
                                  batch_size=batch_size,
                                  augment=augment,
                                  hyperparameters=hyp,
                                  cache_images=cache)
    batch_size = min(batch_size, len(dataset))
    num_workers = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            sampler=sampler,
                            pin_memory=True,
                            collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class LoadImagesAndLabels(Dataset):
    def __init__(self, ds_path, img_size=640, batch_size=16, augment=False, hyperparameters=None, cache_images=False):
        self.img_size = img_size
        self.batch_size = batch_size
        self.cache_images = cache_images
        self.augment = augment
        self.hyp = hyperparameters
        self.label_files = []

        img_folder = Path(ds_path) / "image"
        self.img_files = list(img_folder.glob("*"))

        for img_file in tqdm(self.img_files):
            text_file = (Path(ds_path) / "yolo_annos" / img_file.stem).with_suffix(".txt")
            self.label_files.append(text_file)

        self.labels = []
        for label_file in tqdm(self.label_files):
            with open(label_file) as f:
                labels = [value.split() for value in f.read().strip().splitlines() if len(value)]
                labels = np.array(labels, dtype=np.float32)
                self.labels.append(labels)

        if self.cache_images:
            self.imgs = [self.__reshape_image(Image.open(filepath), desired_size=self.img_size) for filepath in
                         self.img_files]

        self.amount_of_images = len(self.img_files)
        self.indices = range(self.amount_of_images)

    def __reshape_image(self, image, padding=0, desired_size=640):
        old_size = image.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im_P = image.resize(new_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (desired_size + (padding * 2), desired_size + (padding * 2)), (120, 120, 120))
        offset_x = ((desired_size + (padding * 2)) - new_size[0]) // 2
        offset_y = ((desired_size + (padding * 2)) - new_size[1]) // 2
        random_offset_x = random.randint(0, offset_x)
        random_offset_y = random.randint(0, offset_y)
        new_im.paste(im_P, (random_offset_x, random_offset_y))
        im = np.array(new_im)
        return im, im_P.size, random_offset_x, random_offset_y

    @staticmethod
    def collate_fn(batch):
        imgs, labels, = zip(*batch)  # transposed
        for i, label in enumerate(labels):
            label[:, 0] = i  # add target image index for build_targets()
        return torch.stack(imgs, 0), torch.cat(labels, 0)

    @staticmethod
    def batch_collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = imgs[0]
        labels = labels[0]
        return torch.stack(imgs, 0) / 255, torch.from_numpy(labels)

    def get_image_and_labels(self, img_index, position=None, desired_size=640):
        if not self.cache_images:
            img_source = Image.open(self.img_files[img_index])
            img, (reshaped_width, reshaped_height), random_offset_x, random_offset_y = self.__reshape_image(img_source,
                                                                                                            desired_size=desired_size)
        else:
            img, (reshaped_width, reshaped_height), random_offset_x, random_offset_y = self.imgs[img_index]
        img_source_labels = self.labels[img_index]

        resized_labels = []

        for label in img_source_labels:
            label_class = label[0]
            rescaled_x = (label[1] * reshaped_width + random_offset_x) / desired_size
            rescaled_y = (label[2] * reshaped_height + random_offset_y) / desired_size
            rescaled_w = label[3] * reshaped_width / desired_size
            rescaled_h = label[4] * reshaped_height / desired_size
            resized_label = [label_class, rescaled_x, rescaled_y, rescaled_w, rescaled_h]
            if position is not None:
                resized_label.insert(0, float(position))
            resized_labels.append(resized_label)

        return img, np.array(resized_labels, dtype=np.float32)

    def batch_images_and_labels(self, batch_index, desired_size=640):
        batch_imgs = []
        batch_labels = []
        for item_index in range(batch_index * self.batch_size, (batch_index + 1) * self.batch_size):
            if item_index > len(self.img_files):
                continue
            img, labels = self.get_image_and_labels(item_index, position=item_index - batch_index * self.batch_size,
                                                    desired_size=desired_size)
            batch_imgs.append(img)
            batch_labels.append(labels)
        return batch_imgs, np.concatenate(batch_labels)

    def create_mosaic(self, index):
        small_size = self.img_size
        big_size = small_size * 2
        center_x, center_y = (small_size, small_size)
        indixes = [index] + random.choices(self.indices, k=3)
        offsets = ((0, 0), (0, small_size), (small_size, 0), (small_size, small_size))

        quad_image = np.full((big_size, big_size, 3), 120, dtype=np.uint8)
        quad_labels = []

        for i, index in enumerate(indixes):
            img, labels = self.get_image_and_labels(index, desired_size=small_size)

            for label in labels:
                label_class = label[0]
                fragment_x = (label[1] * small_size + offsets[i][1]) / big_size
                fragment_y = (label[2] * small_size + offsets[i][0]) / big_size
                fragment_w = label[3] * small_size / big_size
                fragment_h = label[4] * small_size / big_size

                new_label = [label_class, fragment_x, fragment_y, fragment_w, fragment_h]
                quad_labels.append(new_label)
            quad_image[offsets[i][0]:center_y + offsets[i][0], offsets[i][1]:center_x + offsets[i][1]] = img

        quad_labels = np.array(quad_labels, dtype=np.float32)
        quad_labels[:, 1:] = xywhn2xyxy(quad_labels[:, 1:], w=big_size, h=big_size)
        quad_image, quad_labels = random_augmentation(quad_image, quad_labels,
                                                      img_size=small_size,
                                                      degrees=self.hyp['degrees'],
                                                      translate=self.hyp['translate'],
                                                      scale=self.hyp['scale'],
                                                      shear=self.hyp['shear'],
                                                      perspective=self.hyp['perspective'])

        return quad_image, quad_labels

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        make_mosaic = self.augment and random.random() < self.hyp["mosaic"]
        if make_mosaic:
            img, labels = self.create_mosaic(index)
            number_labels = len(labels)

            if number_labels:
                labels[:, 1:] = xyxy2xywhn(labels[:, 1:])
        else:
            img, labels = self.get_image_and_labels(index)  # 640

            if self.augment:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:])
                img, labels = random_augmentation(img,
                                                  labels,
                                                  degrees=self.hyp['degrees'],
                                                  translate=self.hyp['translate'],
                                                  scale=self.hyp['scale'],
                                                  shear=self.hyp['shear'],
                                                  perspective=self.hyp['perspective'])

                number_labels = len(labels)

                if number_labels:
                    labels[:, 1:] = xyxy2xywhn(labels[:, 1:])

        if self.augment:
            augment_hsv(img, h_gain=self.hyp['hsv_h'], s_gain=self.hyp['hsv_s'], v_gain=self.hyp['hsv_v'])

            if random.random() < self.hyp['flipud']:
                img = np.flipud(img)

                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]

            if random.random() < self.hyp['fliplr']:
                img = np.fliplr(img)

                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

        labels_torch = torch.zeros((len(labels), 6))
        if len(labels):
            labels_torch[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_torch
