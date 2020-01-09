import tensorflow as tf
from tensorflow.python.training import moving_averages

import cfg
from utils import int_quantize, uint_quantize, _variable_on_cpu, _variable_with_weight_decay


def batch_norm_for_conv(x, phase_train, scope='bn'):
    channels = x.shape.as_list()[3]
    with tf.variable_scope(scope):
        gamma = _variable_on_cpu('gamma', [channels, ], tf.constant_initializer(1.0), dtype='float32')
        beta = _variable_on_cpu('beta', [channels, ], tf.constant_initializer(0.0), dtype='float32')
        moving_mean = _variable_on_cpu('moving_mean', [channels, ], dtype='float32',
                                       initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = _variable_on_cpu('moving_variance', [channels, ], dtype='float32',
                                           initializer=tf.zeros_initializer(), trainable=False)
        tf.add_to_collection('params', gamma)
        tf.add_to_collection('params', beta)
        tf.add_to_collection('params', moving_mean)
        tf.add_to_collection('params', moving_variance)

        if not phase_train:
            normed_x, _, _ = tf.nn.fused_batch_norm(x, gamma, beta,
                                                    mean=moving_mean, variance=moving_variance,
                                                    is_training=False, epsilon=cfg.bn_eps)
        else:
            normed_x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta,
                                                                     is_training=True, epsilon=cfg.bn_eps)

            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, cfg.bn_momentum)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_var, cfg.bn_momentum)

            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

        return normed_x, [x, moving_mean, moving_variance, beta, gamma]


def conv_bn_relu(x, x_float, out_channels, ksize, stride=1, groups=1,
                 qweight=False, qactivation=False,
                 padding='SAME', scale=None,
                 has_bn=True, has_relu=True, phase_train=False,
                 scope=None):
    node = {'input': x}

    cfg_node = {'name': scope,
                'type': 'Conv2D',
                'out': out_channels,
                'in': 0,
                'ksize': ksize,
                'stride': stride,
                'groups': groups,
                'padding': padding,
                'active': has_relu,
                'bn': has_bn}

    with tf.variable_scope(scope):
        in_channels = x.shape.as_list()[3]
        cfg_node['in'] = in_channels

        assert in_channels % groups == 0 and out_channels % groups == 0
        shape = [ksize, ksize, in_channels // groups, out_channels]
        kernel = _variable_with_weight_decay('W', shape)
        tf.add_to_collection('params', kernel)
        node['W'] = kernel
        if qweight:
            num_bits = scale[scope]['w_bit']
            kernel, _ = int_quantize(kernel, scale[scope]['W'], num_bits=num_bits, phase_train=phase_train)

        if groups == 1:
            f = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding=padding)
        else:
            if out_channels == groups and in_channels == groups:
                f = tf.nn.depthwise_conv2d(x,
                                           tf.transpose(kernel, (0, 1, 3, 2)),
                                           [1, stride, stride, 1],
                                           padding=padding)
            else:
                kernel_list = tf.split(kernel, groups, axis=3)
                x_list = tf.split(x, groups, axis=3)
                f = tf.concat(
                    [tf.nn.conv2d(x_list[i], kernel_list[i], [1, stride, stride, 1], padding=padding)
                     for i in range(groups)], axis=3)

        if has_bn:
            f, bn_info = batch_norm_for_conv(f, phase_train)
            _, moving_mean, moving_variance, beta, gamma = bn_info
            node['mean'] = moving_mean
            node['var'] = moving_variance
            node['beta'] = beta
            node['gamma'] = gamma
        else:
            biases = _variable_on_cpu('b', out_channels, tf.constant_initializer(0.0))
            tf.add_to_collection('params', biases)
            node['b'] = biases

            f = tf.nn.bias_add(f, biases)

        f_float = None
        if qactivation:
            num_bits = scale[scope]['a_bit']
            if has_relu:
                f, f_float = uint_quantize(f, scale[scope]['output'], num_bits=num_bits, phase_train=phase_train)
            else:
                f, f_float = int_quantize(f, scale[scope]['output'], num_bits=num_bits, phase_train=phase_train)
        else:
            if has_relu:
                f = tf.nn.relu6(f)
            f_float = f
        node['output'] = f
        print(scope, f.shape)

        tf.add_to_collection('nodes', node)
        tf.add_to_collection('cfg_nodes', cfg_node)

        return f, f_float


def add(x, x_float, y, y_float, phase_train, has_relu=False, scale=None, qactivation=False, scope=None):
    if cfg.add_with_float:
        f = x_float + y_float
    else:
        f = x + y
    if qactivation:
        num_bits = scale[scope]['a_bit']
        if has_relu:
            f, f_float = uint_quantize(f, scale[scope]['output'], num_bits=num_bits, phase_train=phase_train)
        else:
            f, f_float = int_quantize(f, scale[scope]['output'], num_bits=num_bits, phase_train=phase_train)
    else:
        if has_relu:
            f = tf.nn.relu6(f)
        f_float = f
    node = {'input': [x, y],
            'output': f}
    cfg_node = {'name': scope,
                'type': 'Add'}
    tf.add_to_collection('nodes', node)
    tf.add_to_collection('cfg_nodes', cfg_node)

    return f, f_float
