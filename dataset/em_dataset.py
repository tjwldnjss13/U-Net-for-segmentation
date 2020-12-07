import os
import cv2 as cv
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image


class EMDataset(data.Dataset):
    def __init__(self, root, is_train=True, shuffle=False):
        self.root = root
        if is_train:
            self.images = self.get_images('train')
            self.labels = self.get_images('label')
            self.images, self.labels = EMDataset.augment_images(self.images, self.labels)
        else:
            self.images = self.get_images('test')

        if shuffle:
            datas = list(zip(self.images, self.labels))
            np.random.shuffle(datas)
            images, labels = zip(*datas)
            self.images, self.labels = list(images), list(labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)

    def get_images(self, dir):
        imgs_dir = os.path.join(self.root, dir)
        imgs_fn = os.listdir(imgs_dir)
        imgs = []

        for i, img_fn in enumerate(imgs_fn):
            img_pth = os.path.join(imgs_dir, img_fn)
            img = Image.open(img_pth)
            img = np.array(img)
            imgs.append(img)

        return imgs

    @staticmethod
    def augment_images(images, labels):
        imgs_rot, labels_rot = EMDataset.augment_images_rotate(images, labels, [90, 180, 270])

        images += imgs_rot
        labels += labels_rot

        imgs_shift, labels_shift = EMDataset.augment_images_shift(images, labels, [1, 2, 3, 4, 5])

        images += imgs_shift
        labels += labels_shift

        return images, labels

    @staticmethod
    def augment_images_shift(images, labels, shift_distances):
        imgs_shift = []
        labels_shift = []
        n_data = len(images)

        for i in range(n_data):
            for d in shift_distances:
                if d <= 5:
                    img, label = images[i], labels[i]

                    imgs_shift.append(EMDataset.shift_right_image(img, d))
                    imgs_shift.append(EMDataset.shift_left_image(img, d))
                    imgs_shift.append(EMDataset.shift_up_image(img, d))
                    imgs_shift.append(EMDataset.shift_down_image(img, d))

                    labels_shift.append(EMDataset.shift_right_image(label, d))
                    labels_shift.append(EMDataset.shift_left_image(label, d))
                    labels_shift.append(EMDataset.shift_up_image(label, d))
                    labels_shift.append(EMDataset.shift_down_image(label, d))

        return imgs_shift, labels_shift

    @staticmethod
    def shift_right_image(image, shift_distance):
        d = shift_distance
        img_shift = np.zeros(image.shape)

        for i in range(d):
            img_shift[:, i] = image[:, 0]

        img_temp = image[:, :-d]
        img_shift[:, d:] = img_temp

        return img_shift

    @staticmethod
    def shift_left_image(image, shift_distance):
        d = shift_distance
        img_shift = np.zeros(image.shape)

        for i in range(d):
            img_shift[:, -i] = image[:, -1]

        img_temp = image[:, d:]
        img_shift[:, :-d] = img_temp

        return img_shift

    @staticmethod
    def shift_up_image(image, shift_distance):
        d = shift_distance
        img_shift = np.zeros(image.shape)

        for i in range(d):
            img_shift[-i, :] = image[-1, :]

        img_temp = image[d:, :]
        img_shift[:-d, :] = img_temp

        return img_shift

    @staticmethod
    def shift_down_image(image, shift_distance):
        d = shift_distance
        img_shift = np.zeros(image.shape)

        for i in range(d):
            img_shift[i, :] = image[0, :]

        img_temp = image[:-d, :]
        img_shift[d:, :] = img_temp

        return img_shift


    @staticmethod
    def augment_images_rotate(images, labels, degrees):
        imgs_rot = []
        labels_rot = []
        n_data = len(images)

        for i in range(n_data):
            for r in degrees:
                if r != 0:
                    img, label = images[i], labels[i]

                    h, w = img.shape[0], img.shape[1]
                    M = cv.getRotationMatrix2D((w / 2, h / 2), r, 1)

                    imgs_rot.append(cv.warpAffine(img, M, (w, h)))
                    labels_rot.append(cv.warpAffine(label, M, (w, h)))

        return imgs_rot, labels_rot

    @staticmethod
    def augment_images_elastic_deform(images):
        pass