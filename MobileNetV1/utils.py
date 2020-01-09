import tensorflow as tf
import math

import cfg


def int_quantize(x, scale_factor, num_bits=8, phase_train=False):
    x_float = x
    max_int = 2 ** (num_bits - 1) - 1

    if phase_train:
        x_scale = x / scale_factor
        x_int = tf.stop_gradient(tf.round(x_scale) - x_scale) + x_scale
    else:
        x_int = tf.round(x / scale_factor)
    x_clip = tf.clip_by_value(x_int, -max_int - 1, max_int)
    return x_clip * scale_factor, x_float


def uint_quantize(x, scale_factor, num_bits=8, phase_train=False):
    x_float = x
    max_int = 2 ** num_bits - 1

    if phase_train:
        x_scale = x / scale_factor
        x_int = tf.stop_gradient(tf.round(x_scale) - x_scale) + x_scale
    else:
        x_int = tf.round(x / scale_factor)
    x_clip = tf.clip_by_value(x_int, 0, max_int)
    return x_clip * scale_factor, x_float


def compute_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
        return fan_in, fan_out
    elif len(shape) == 4:
        ksize = shape[0] * shape[1]
        fan_in = shape[2] * ksize
        fan_out = shape[3] * ksize
        return fan_in, fan_out
    else:
        raise NotImplementedError


def _variable_on_cpu(name, shape, initializer, trainable=True, dtype=cfg.dtype):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the Variable
      shape: list of ints
      initializer: initializer of Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, dtype=cfg.dtype):
    fan_in, fan_out = compute_fans(shape)
    stddev = math.sqrt(1.0 / fan_in)
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev, dtype=cfg.dtype),
                           dtype=dtype)
    if cfg.weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), cfg.weight_decay, name='weight_loss')
        if cfg.dtype == 'float16':
            weight_decay = tf.to_float(weight_decay)
        tf.add_to_collection('losses', weight_decay)
    return var
