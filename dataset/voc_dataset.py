import os
import sys
import random
import torch
import torch.utils.data as data
import xml.etree.ElementTree as Et

from PIL import Image
from xml.etree.ElementTree import Element, ElementTree


class VOCDataset(data.Dataset):
    def __init__(self, root, segmentation=False, is_validation=False, valid_split=.3, shuffle=False, transforms=None, is_categorical=True):
        self.root = root
        self.segmentation = segmentation
        self.is_validation = is_validation
        self.valid_split = valid_split
        self.transforms = transforms
        self.class_dict = {'background': 0, 'aerplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8,
                           'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                           'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20, 'ambigious': 255}
        if segmentation:
            self.filenames = self.make_seg_image_list()
            if shuffle:
                random.shuffle(self.filenames)
        if not segmentation:
            self.annotation = self.make_ann_list()
            if shuffle:
                random.shuffle(self.annotation)

    def __getitem__(self, idx):
        if self.segmentation:
            img_dir = os.path.join(self.root, 'JPEGImages')
            img_pth = os.path.join(img_dir, self.filenames[idx])
            img = Image.open(img_pth)

            gt_dir = os.path.join(self.root, 'SegmentationClass')
            gt_pth = os.path.join(gt_dir, self.filenames[idx])
            gt = Image.open(gt_pth)

            if self.transforms is not None:
                img = self.transforms(img)
                gt = self.transforms(gt)
            else:
                img = torch.as_tensor(img, dtype=torch.float64)
                gt = torch.as_tensor(gt, dtype=torch.float64)

            return img, gt
        else:
            img_dir = os.path.join(self.root, 'JPEGImages')
            ann = self.annotation[idx]
            img_fn = ann.find('filename').text
            img_path = os.path.join(img_dir, img_fn)
            img = Image.open(img_path).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            else:
                img = torch.as_tensor(img, dtype=torch.float64)

            obj = ann.findall('object')[0]
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            label = self.class_dict[name] if name in self.class_dict.keys() else 0
            if self.is_categorical:
                label = self.to_categorical(label, 20)
            bbox = [xmin, ymin, xmax, ymax]

            label = torch.as_tensor(label, dtype=torch.int64)
            bbox = torch.as_tensor(bbox, dtype=torch.float32)

            ann = {'label': label, 'bbox': bbox}

            return img, ann

    def __len__(self):
        return len(self.annotation)

    def make_ann_list(self):
        ann_dir = os.path.join(self.root, 'Annotations')
        anns_fn = os.listdir(ann_dir)

        anns_ = []
        for ann_fn in anns_fn:
            ann_path = os.path.join(ann_dir, ann_fn)
            ann = open(ann_path, 'r')
            tree = Et.parse(ann)
            root_ = tree.getroot()
            anns_.append(root_)
            ann.close()

        if self.is_validation:
            anns = anns_[int(len(anns_) * (1 - self.valid_split)):]
        else:
            anns = anns_[:int(len(anns_) * (1 - self.valid_split))]

        print('Annotations loaded!')

        return anns

    def make_seg_image_list(self):
        fn_txt_dir = os.path.join(self.root, 'ImageSets', 'Segmentation')

        if self.is_validation:
            fn_txt_pth = os.path.join(fn_txt_dir, 'train.txt')
        else:
            fn_txt_pth = os.path.join(fn_txt_dir, 'val.txt')

        with open(fn_txt_pth) as file:
            fns = file.readlines()

        for i in range(len(fns)):
            fns[i] = str.strip(fns[i])

        return fns


    @staticmethod
    def to_categorical(label, n_class):
        label_ = [0 for _ in range(n_class)]
        label_[label - 1] = 1

        return label_


def collate_fn(batch):
    images = [item[0] for item in batch]
    anns = [item[1] for item in batch]

    return [images, anns]


if __name__ == '__main__':
    root_dir = 'C://DeepLearningData/VOC2012/'
    train_dset = VOCDataset(root_dir, False)
    valid_dset = VOCDataset(root_dir, True)

    print(len(train_dset))
    print(len(valid_dset))
