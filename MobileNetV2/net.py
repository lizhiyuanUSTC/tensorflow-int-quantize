import tensorflow as tf
import pickle
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm

import cfg
from layer import conv_bn_relu, add
from quantize_model import prepare_calibrate_imgs, find_weight_scale, find_feature_map_scale


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def create_separable_conv(x, x_float, out_channels, ksize, stride=1,
                          qweight=False, qactivation=False, scale=None,
                          has_bn=True, has_relu=True, phase_train=False,
                          block_id=0, ):
    in_channels = x.shape.as_list()[3]
    depthwise_filters = in_channels
    pointwise_filters = _make_divisible(out_channels, 8)
    prefix = 'block_{}_'.format(block_id)

    f = x
    f_float = x_float
    if block_id:
        # Expand
        f, f_float = conv_bn_relu(x, x_float, in_channels * 6, 1, stride=1, qweight=qweight,
                                  qactivation=qactivation, padding='SAME', scale=scale,
                                  has_bn=has_bn, has_relu=has_relu, phase_train=phase_train,
                                  scope=prefix + 'expand')
        depthwise_filters = in_channels * 6
    f, f_float = conv_bn_relu(f, f_float, depthwise_filters, ksize, stride=stride, qweight=qweight,
                              qactivation=qactivation,
                              padding='SAME', groups=depthwise_filters, scale=scale,
                              has_bn=has_bn, has_relu=has_relu, phase_train=phase_train,
                              scope=prefix + 'depthwise')
    f, f_float = conv_bn_relu(f, f_float, pointwise_filters, 1, stride=1, qweight=qweight, qactivation=qactivation,
                              padding='SAME', scale=scale, has_bn=has_bn, has_relu=False, phase_train=phase_train,
                              scope=prefix + 'project')

    if in_channels == pointwise_filters and stride == 1:
        f, f_float = add(f, f_float, x, x_float, phase_train, qactivation=qactivation, scope=prefix + 'add',
                         scale=scale)
    return f, f_float


def inference(images, phase_train=False, has_bn=True, image_norm=True, alpha=1.0,
              qactivation=False, qweight=False, scale=None):
    images = tf.cast(images, dtype=cfg.dtype)
    if image_norm:
        mean = np.reshape(np.array(cfg.image_mean), (1, 1, 1, 3))
        std = np.reshape(np.array(cfg.image_std), (1, 1, 1, 3))
        images = (images - mean) / std
    else:
        images = images - 128

    first_block_filters = _make_divisible(32 * alpha, 8)
    f, f_float = conv_bn_relu(images, None, first_block_filters, 3, 2, qweight=qweight, qactivation=qactivation,
                              scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, scope=cfg.first_conv_name)
    f, f_float = create_separable_conv(f, f_float, int(16 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=0)

    f, f_float = create_separable_conv(f, f_float, int(24 * alpha), 3, 2, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=1)
    f, f_float = create_separable_conv(f, f_float, int(24 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=2)

    f, f_float = create_separable_conv(f, f_float, int(32 * alpha), 3, 2, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=3)
    f, f_float = create_separable_conv(f, f_float, int(32 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=4)
    f, f_float = create_separable_conv(f, f_float, int(32 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=5)

    f, f_float = create_separable_conv(f, f_float, int(64 * alpha), 3, 2, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=6)
    f, f_float = create_separable_conv(f, f_float, int(64 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=7)
    f, f_float = create_separable_conv(f, f_float, int(64 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=8)
    f, f_float = create_separable_conv(f, f_float, int(64 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=9)

    f, f_float = create_separable_conv(f, f_float, int(96 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=10)
    f, f_float = create_separable_conv(f, f_float, int(96 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=11)
    f, f_float = create_separable_conv(f, f_float, int(96 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=12)

    f, f_float = create_separable_conv(f, f_float, int(160 * alpha), 3, 2, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=13)
    f, f_float = create_separable_conv(f, f_float, int(160 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=14)
    f, f_float = create_separable_conv(f, f_float, int(160 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=15)

    f, f_float = create_separable_conv(f, f_float, int(320 * alpha), 3, 1, qweight=qweight, qactivation=qactivation,
                                       scale=scale,
                                       has_bn=has_bn, has_relu=True, phase_train=phase_train, block_id=16)

    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    f, f_float = conv_bn_relu(f, f_float, last_block_filters, 1, 1, qweight=qweight, qactivation=qactivation,
                              scale=scale,
                              has_bn=has_bn, has_relu=True, phase_train=phase_train, scope='conv_1')

    f, f_float = conv_bn_relu(f, f_float, 1000, 1, stride=1, padding='SAME', qweight=qweight, qactivation=False,
                              scale=scale,
                              has_bn=False, has_relu=False, phase_train=phase_train, scope='prediction')
    f = tf.reduce_mean(f, axis=[1, 2], keepdims=False)
    if cfg.dtype == 'float16':
        f = tf.cast(f, dtype='float32')

    return f
