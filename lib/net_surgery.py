import os
import tensorflow as tf
import pickle
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import cfg
from net import inference
from quantize_model import prepare_calibrate_imgs, find_weight_scale, find_feature_map_scale
import dataset


def fix_input(w, b, fix_on_bn):
    mean = np.array(cfg.image_mean, dtype=np.float32)
    std = np.array(cfg.image_std, dtype=np.float32)
    w = w / np.reshape(std, (1, 1, -1, 1))
    k_h, k_w, _, _ = w.shape

    graph = tf.Graph()
    with graph.as_default():
        mean = tf.constant(mean)
        mean = tf.reshape(mean, (1, 1, 1, 3))
        mean = tf.tile(mean, (1, k_h, k_w, 1))
        conv_mean = tf.nn.conv2d(mean - 128, w, strides=[1, 1, 1, 1], padding='VALID')
        if fix_on_bn:
            b = b + tf.squeeze(conv_mean)
        else:
            b = b - tf.squeeze(conv_mean)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            b = sess.run(b)
    return w, b


def fix_input_params():
    graph = tf.Graph()
    with graph.as_default():
        images = np.random.rand(1, 224, 224, 3)
        inference(images, False, image_norm=False, has_bn=True)
        nodes = tf.get_collection('nodes')
        cfg_nodes = tf.get_collection('cfg_nodes')
        node = None
        cfg_node = None
        for i in range(len(nodes)):
            name = cfg_nodes[i]['name']
            if cfg_nodes[i]['type'] == 'Conv2D' and cfg.first_conv_name == name:
                node = nodes[i]
                cfg_node = cfg_nodes[i]
                break

        model_checkpoint_path = 'log/model_dump/model.ckpt'
        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_checkpoint_path)

            _node = sess.run(node)

            if cfg_node['bn']:
                W = _node['W']
                mean = _node['mean']
                W, mean = fix_input(W, mean, fix_on_bn=True)
                sess.run(tf.assign(node['W'], W))
                sess.run(tf.assign(node['mean'], mean))
            else:
                W = _node['W']
                b = _node['b']
                W, b = fix_input(W, b, fix_on_bn=False)
                sess.run(tf.assign(node['W'], W))
                sess.run(tf.assign(node['b'], b))

            saver.save(sess, 'log/model_dump/model_fix_input.ckpt')


def find_quantize_scale(model_checkpoint_path):
    graph = tf.Graph()
    with graph.as_default():
        images = prepare_calibrate_imgs()

        _ = inference(images, False, has_bn=True, image_norm=False)
        nodes = tf.get_collection('nodes')
        cfg_nodes = tf.get_collection('cfg_nodes')
        cfg_nodes = find_connect(nodes, cfg_nodes)

        saver = tf.train.Saver(tf.get_collection('params'))
        scale_dict = OrderedDict()

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_checkpoint_path)

            if os.path.exists('log/scale'):
                with open('log/scale', 'rb') as f:
                    scale_dict = pickle.load(f)

            nodes = sess.run(nodes)

            for i in tqdm(range(len(nodes))):
                name = cfg_nodes[i]['name']
                node = nodes[i]
                scale_dict[name] = {}

                if cfg_nodes[i]['type'] == 'Conv2D':
                    weight = node['W']
                    cfg_nodes[i]['W'] = weight
                    num_bit = cfg.w_bit
                    if cfg_nodes[i]['name'] == cfg.first_conv_name:
                        num_bit = 8
                    scale = find_weight_scale(weight, num_bit=num_bit)
                    scale_dict[name]['W'] = scale
                    scale_dict[name]['w_bit'] = num_bit
                    cfg_nodes[i]['scale_W'] = scale

                    if cfg_nodes[i]['bn']:
                        cfg_nodes[i]['mean'] = node['mean']
                        cfg_nodes[i]['var'] = node['var']
                        cfg_nodes[i]['gamma'] = node['gamma']
                        cfg_nodes[i]['beta'] = node['beta']
                    else:
                        biases = node['b']
                        cfg_nodes[i]['b'] = biases

                outputs = node['output']
                scale = find_feature_map_scale(outputs, num_bit=cfg.a_bit)
                scale_dict[name]['output'] = scale
                scale_dict[name]['a_bit'] = cfg.a_bit
                cfg_nodes[i]['scale_output'] = scale

            with open('log/scale', 'wb') as f:
                pickle.dump(scale_dict, f)

            with open('log/cfg_nodes.pkl', 'wb') as f:
                pickle.dump(cfg_nodes, f)


def bn_ema(model_checkpoint_path='log/model_dump/model_fix_input.ckpt',
           qweight=True, qactivation=True):

    with open('log/scale', 'rb') as f:
        scale = pickle.load(f)

    graph = tf.Graph()
    with graph.as_default():
        iterator = dataset.make_train_dataset()
        images, labels = iterator.get_next()
        inference(images, True, has_bn=True, image_norm=False,
                  qweight=qweight, qactivation=qactivation, scale=scale)

        update_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_checkpoint_path)

            for i in tqdm(range(100)):
                sess.run(update_bn)

            saver.save(sess, 'log/model_dump/model_fix_input_bn_ema.ckpt')


def find_connect(nodes, cfg_nodes):
    for i in range(len(cfg_nodes)):
        if cfg_nodes[i]['name'] == cfg.first_conv_name:
            cfg_nodes[i]['input_layer'] = 'image'
        if cfg_nodes[i]['type'] == 'Conv2D':
            input = nodes[i]['input']
            for j in range(len(cfg_nodes)):
                output = nodes[j]['output']
                if output.name == input.name:
                    cfg_nodes[i]['input_layer'] = cfg_nodes[j]['name']
                    break
        elif cfg_nodes[i]['type'] == 'Add':
            input = nodes[i]['input']
            cfg_nodes[i]['input_layer'] = []
            for _input in input:
                for j in range(len(cfg_nodes)):
                    output = nodes[j]['output']
                    if output.name == _input.name:
                        cfg_nodes[i]['input_layer'].append(cfg_nodes[j]['name'])
                        break
        else:
            raise NotImplementedError

    for node in cfg_nodes:
        print(node['name'], node['input_layer'])

    return cfg_nodes
