import glob
import math
import pickle
import random
import re

import tensorflow as tf
import keras as K
import os
import numpy as np
import cv2
from keras.utils import Sequence
import logging
from sklearn.utils.class_weight import compute_class_weight


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

file_path = os.path.dirname(__file__)
SEPARATOR = os.path.sep


class Generator:
    def __init__(self, ds_folder, batch_size=64, w=64, h=128, val_to_train=0.15,
                 augmenter=None, preprocessor=None, with_camera_info=False, camera_pattern='C(.*)T', cat_based=True):
        cats_folders = list(sorted([name for name in os.listdir(ds_folder)
                                    if os.path.isdir(ds_folder + SEPARATOR + name)]))

        self.cat_based = cat_based
        self.camera_pattern = camera_pattern
        self.with_camera_info = with_camera_info
        self.random_state = np.random.RandomState(1234)
        self.ds_imgs = []
        self.ds_labels = []
        logger.debug("start init of dataset")

        self.name_to_idx = {}
        self.idx_to_name = {}
        self.classes_elements_num = {}
        self.max_classes_elements_num = 0
        cats_folders = list(cats_folders)
        random.Random(1234).shuffle(cats_folders)
        for i, cat in enumerate(cats_folders):
            cnt = 0
            for img_path in glob.glob(ds_folder + SEPARATOR + str(cat) + SEPARATOR + '*.jpg') + \
                            glob.glob(ds_folder + SEPARATOR + str(cat) + SEPARATOR + '*.png'):
                self.ds_imgs.append(img_path)
                self.ds_labels.append(i)
                cnt += 1

            self.max_classes_elements_num = max([self.max_classes_elements_num, cnt])
            self.name_to_idx[cat] = i
            self.idx_to_name[i] = cat
            self.classes_elements_num[i] = cnt

        self.ds_imgs = np.asarray(self.ds_imgs)
        self.ds_labels = np.asarray(self.ds_labels)
        self.all_length = self.ds_imgs.shape[0]
        self.batch_size = batch_size
        self.w = w
        self.h = h
        self.val_to_train = val_to_train
        self.cats_num = len(cats_folders)

        self.augmenter = augmenter
        self.preprocessor = preprocessor

        all_indices = np.arange(self.ds_imgs.shape[0])
        if not self.cat_based:
            self.random_state.shuffle(all_indices)
            self.train_indices, self.val_indices = np.split(all_indices,
                                                            [int((1 - val_to_train) * self.all_length)])
        else:
            cats = np.unique(self.ds_labels)
            cats_train = self.random_state.choice(cats, int(cats.shape[0] * (1 - val_to_train)), replace=False)
            self.train_indices = all_indices[np.isin(self.ds_labels, cats_train)]
            self.val_indices = all_indices[~np.isin(self.ds_labels, cats_train)]
            logger.debug("total %s cats in train", cats_train.shape[0])

        self.train_length = self.train_indices.shape[0]
        self.val_length = self.val_indices.shape[0]
        logger.debug("dataset inited")
        logger.debug("total %s img in train", self.train_length)
        logger.debug("total %s img in val", self.val_length)
        logger.debug("total %s categories", self.cats_num)

    def obtain_class_weights(self):
        # res = {}
        # for item, val in self.classes_elements_num.items():
        #     res[item] = float(self.max_classes_elements_num) / (val + 1)
        #
        class_weights = compute_class_weight('balanced', np.array(range(self.cats_num)), self.ds_labels)
        res = dict(enumerate(class_weights))
        return res

    class InnerGenerator(Sequence):

        def __init__(self, outer, indexes, is_train):
            self.indixes = indexes
            self.outer = outer
            self.length = self.indixes.shape[0]
            self.is_train = is_train
            self.with_camera_info = self.outer.with_camera_info

        def on_epoch_end(self):
            self.outer.random_state.shuffle(self.indixes)

        def __getitem__(self, idx):
            try:
                inds = self.indixes[idx * self.outer.batch_size:(idx + 1) * self.outer.batch_size]
                images = self.outer.ds_imgs[inds]
                labels = self.outer.ds_labels[inds]

                labels_y = []
                imgs = []
                cameras = []
                for i in range(inds.shape[0]):
                    img = cv2.imread(images[i])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.outer.w, self.outer.h))

                    imgs.append(img)
                    labels_y.append(K.utils.to_categorical([int(labels[i])], num_classes=self.outer.cats_num)[0])
                    if self.outer.with_camera_info:
                        cameras.append(re.search(self.outer.camera_pattern, images[i]).group(1))

                imgs = np.array(imgs)
                if self.is_train and self.outer.augmenter is not None:
                    imgs = self.outer.augmenter(imgs)

                if self.outer.preprocessor is not None:
                    imgs = self.outer.preprocessor(imgs)

                labels_x = imgs
                labels_y = np.asarray(labels_y)
                cameras = np.asarray(cameras)

                if self.outer.with_camera_info:
                    return labels_x, labels_y, cameras
                else:
                    return labels_x, labels_y
            except Exception as e:
                logger.exception("error on generation")

        def __len__(self):
            return int(math.ceil(float(self.length) / self.outer.batch_size))

    def make_train_generator(self):
        return Generator.InnerGenerator(self, self.train_indices, True)

    def make_val_generator(self):
        return Generator.InnerGenerator(self, self.val_indices, False)
