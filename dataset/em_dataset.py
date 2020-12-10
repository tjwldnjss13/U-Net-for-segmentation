import os
import cv2 as cv
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

        self.images = numpy_to_tensor_x(self.images)
        self.labels = numpy_to_tensor_y(self.labels)

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
            img = np.array(img) / 255.
            imgs.append(img)

        return imgs

    @staticmethod
    def augment_images(images, labels):
        print(len(images))
        imgs_ed, labels_ed = EMDataset.augment_images_elastic_deformation(images, labels, [3, 5, 7, 9], [1, 2, 3])
        images += imgs_ed
        labels += labels_ed
        print(len(images))

        imgs_rot, labels_rot = EMDataset.augment_images_rotate(images, labels, [90, 180, 270])
        images += imgs_rot
        labels += labels_rot
        print(len(images))

        # imgs_shift, labels_shift = EMDataset.augment_images_shift(images, labels, [1])
        # images += imgs_shift
        # labels += labels_shift
        # print(len(images))

        return images, labels

    @staticmethod
    def augment_images_shift(images, labels, shift_distances):
        print('Shift augmentation...')
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
        print('Rotate augmentation...')
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
    def augment_images_elastic_deformation(images, labels, sigmas, alphas):
        print('Elatstic deformation augmentation...')
        imgs_ed = []
        labels_ed = []
        n_data = len(images)

        for i in range(n_data):
            img, label = images[i], labels[i]

            for s in sigmas:
                for a in alphas:
                    img_ed, label_ed = EMDataset.elastic_deformation(img, label, s, a)

                    imgs_ed.append(img_ed)
                    labels_ed.append(label_ed)

        return imgs_ed, labels_ed

    @staticmethod
    def elastic_deformation(img, label, sigma, alpha):
        img_dst = np.zeros(img.shape)
        label_dst = np.zeros(label.shape)

        # Sampling from Unif(-1, 1)
        dx = np.random.uniform(-1, 1, img.shape[:2])
        dy = np.random.uniform(-1, 1, img.shape[:2])

        # STD of gaussian kernel
        sig = sigma

        dx_gauss = cv.GaussianBlur(dx, (7, 7), sig)
        dy_gauss = cv.GaussianBlur(dy, (7, 7), sig)

        n = np.sqrt(dx_gauss ** 2 + dy_gauss ** 2)  # for normalization

        # Strength of distortion
        alpha = alpha

        ndx = alpha * dx_gauss / n
        ndy = alpha * dy_gauss / n

        indy, indx = np.indices(img.shape[:2], dtype=np.float32)

        map_x = ndx + indx
        map_x = map_x.reshape(img.shape[:2]).astype(np.float32)
        map_y = ndy + indy
        map_y = map_y.reshape(img.shape[:2]).astype(np.float32)

        img_dst = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)
        label_dst = cv.remap(label, map_x, map_y, cv.INTER_LINEAR)

        return img_dst, label_dst


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # print('data: ', data)
    # print('target: ', target)
    return [data, target]


def numpy_to_tensor_x(array):
    for i, a in enumerate(array):
        tensor = torch.as_tensor(a, dtype=torch.float)
        if len(tensor.shape) < 3:
            tensor = tensor.unsqueeze(0)
        array[i] = tensor

    return array


def numpy_to_tensor_y(array):
    for i, a in enumerate(array):
        tensor = torch.zeros((2, 512, 512), dtype=torch.long)
        tensor[0] = torch.as_tensor((a == 0), dtype=torch.long)
        tensor[1] = torch.as_tensor((a == 1), dtype=torch.long)
        # tensor = torch.as_tensor(a, dtype=torch.long).to(device)
        array[i] = tensor

    return array


if __name__ == '__main__':
    root = '../data/'
    dset = EMDataset(root, True, True)

    n_data = len(dset)
    img, label = dset[0]
    print(label.shape)
