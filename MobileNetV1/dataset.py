import tensorflow as tf
import random
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

import cfg


def train_func(image,
               resize=transforms.Resize(cfg.min_size)
               ):
    image = Image.fromarray(image)
    image = resize(image)
    image = np.array(image)
    return image


def train_parse(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3, dct_method='INTEGER_ACCURATE')
    image = tf.py_func(train_func, [image], tf.uint8)
    image = tf.random_crop(image, (cfg.image_size, cfg.image_size, 3))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image.set_shape([cfg.image_size, cfg.image_size, 3])
    return image, label


def val_func(image,
             resize=transforms.Resize(cfg.min_size),
             center_crop=transforms.CenterCrop(cfg.image_size)):
    image = Image.fromarray(image)
    image = resize(image)
    image = center_crop(image)
    image = np.array(image)
    return image


def val_parse(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3, dct_method='INTEGER_ACCURATE')
    image = tf.py_func(val_func, [image], tf.uint8)
    image.set_shape([cfg.image_size, cfg.image_size, 3])
    return image, label


def make_train_dataset(dir=cfg.train_dir,
                       batch_size=cfg.batch_size):
    with open(cfg.train_txt, 'r') as f:
        img_info = f.readlines()
    random.shuffle(img_info)
    filename_list = []
    label_list = []
    for line in img_info:
        line = line.rstrip('\n')
        line = line.split(' ')
        filename_list.append(os.path.join(dir, line[0]))
        label_list.append(int(line[1]))

    filenames = tf.constant(filename_list)
    labels = tf.constant(label_list)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=20 * batch_size)

    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=train_parse, batch_size=batch_size,
        num_parallel_calls=cfg.data_num_threads, drop_remainder=True,
    ))
    dataset = dataset.prefetch(buffer_size=20)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator


def make_val_dataset(dir=cfg.val_dir):
    with open(cfg.val_txt, 'r') as f:
        img_info = f.readlines()

    filename_list = []
    label_list = []
    for line in img_info:
        line = line.rstrip('\n')
        line = line.split(' ')
        filename_list.append(os.path.join(dir, line[0]))
        label_list.append(int(line[1]))

    filenames = tf.constant(filename_list)
    labels = tf.constant(label_list)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=val_parse, batch_size=cfg.eval_batch_size,
        num_parallel_calls=cfg.data_num_threads, drop_remainder=True
    ))
    dataset = dataset.prefetch(buffer_size=5)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator
