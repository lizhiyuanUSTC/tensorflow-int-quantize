from tqdm import tqdm
import tensorflow as tf
import os
import numpy as np
import pickle
from multiprocessing import Pool, Process, Queue
from tqdm import tqdm
import cv2
import random
from collections import OrderedDict

import cfg


def compute_mse_error(x, scale, num_bit, sysmetric=False):
    max_quantized = 2 ** (num_bit - 1) - 1
    min_quantized = -max_quantized - 1
    if np.min(x) >= 0 and sysmetric is False:
        max_quantized = 2 ** num_bit - 1
        min_quantized = 0
    x_quantized = np.round(x / scale)
    x_quantized = np.clip(x_quantized, min_quantized, max_quantized)
    assert x_quantized.min() >= min_quantized and x_quantized.max() <= max_quantized
    x_quantized = x_quantized * scale
    mse = np.power(x_quantized - x, 2)
    return mse


def compute_mse(x, scale, queue, num_bit):
    mse = np.mean(compute_mse_error(x, scale, num_bit))
    queue.put([scale, mse])


def find_best_scale_multi_core(x, num_step=10, num_bit=8):
    x_max_value = np.max(np.abs(x))
    max_quantized = 2 ** (num_bit - 1) - 1
    min_quantized = -max_quantized - 1
    if np.min(x) >= 0:
        max_quantized = 2 ** num_bit - 1
        min_quantized = 0

    clip_value_list = [x_max_value * i / num_step for i in range(1, num_step+1)]
    if np.min(x) < 0:
        clip_value_list += [-x_max_value * i / num_step for i in range(1, num_step + 1)]
    scale_list = []
    mse_list = []

    result_queue = Queue()
    core_list = []
    for clip_value in clip_value_list:
        core = Process(target=compute_mse, args=(x, clip_value / max_quantized, result_queue, num_bit))
        core.start()
        core_list.append(core)

    for core in core_list:
        core.join()

    while result_queue.qsize() > 0:
        scale, mse = result_queue.get()
        scale_list.append(scale)
        mse_list.append(mse)

    mse_list = np.array(mse_list)
    scale_list = np.array(scale_list)

    best_scale = scale_list[np.argmin(mse_list)]
    return best_scale


def find_best_scale_numpy(x, num_step=20, num_bit=8):
    _len = len(x.shape)
    scale_list = []
    x_max_value = np.max(np.abs(x))
    max_quantized = 2 ** (num_bit - 1) - 1
    min_quantized = -max_quantized - 1
    for i in range(1, num_step + 1):
        clip_value = x_max_value * i / num_step
        scale = clip_value / max_quantized
        scale_list.append(scale)
        if np.min(x) < 0:
            scale_list.append(-scale)

    x = np.expand_dims(x, 0)
    scale_list = np.array(scale_list)
    for _ in range(_len):
        scale_list = np.expand_dims(scale_list, -1)
    mse = compute_mse_error(x, scale_list, num_bit, sysmetric=True)
    for _ in range(_len):
        mse = np.mean(mse, 1)
    scale_list = np.squeeze(scale_list)
    scale = scale_list[np.argmin(mse)]
    # print(scale, scale*max_quantized, np.max(x))
    # print(scale, scale * min_quantized, np.min(x))
    return scale


def find_weight_scale(x, num_step=20, num_bit=8):
    scale = []
    for i in range(x.shape[-1]):
        scale.append(find_best_scale_numpy(x[:, :, :, i], num_step=num_step, num_bit=num_bit))
    scale = np.array(scale, dtype=np.float32)
    scale = np.reshape(scale, (1, 1, 1, -1))
    return scale


def find_feature_map_scale(x, num_bit=8, num_step=10):
    scale = find_best_scale_multi_core(x, num_step=num_step, num_bit=num_bit)
    return scale


def prepare_calibrate_imgs(img_dir='/home/lzy/DL_DATA/ILSVRC/Data/CLS-LOC/val/'):
    img_path_list = os.listdir(img_dir)
    random.shuffle(img_path_list)
    img_path_list = img_path_list[:100]
    imgs = []
    for img_path in img_path_list:
        img = cv2.imread(os.path.join(img_dir, img_path))
        img = cv2.resize(img, (224, 224))
        imgs.append(img)
    imgs = np.array(imgs, dtype=np.uint8)
    return imgs
