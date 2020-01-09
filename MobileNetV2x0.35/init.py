import os
import tensorflow as tf
import pickle
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from keras.applications import MobileNetV2

import cfg
from net import inference
from net_surgery import fix_input_params, find_quantize_scale, bn_ema
from tools import train, evaluate


def init():
    network = MobileNetV2(alpha=0.35)
    params = network.get_weights()

    graph = tf.Graph()
    with graph.as_default():
        images = np.random.rand(1, 224, 224, 3)

        inference(images, False)

        model_checkpoint_path = 'log/model_dump/model.ckpt'
        var_list = tf.get_collection('params')
        assert len(var_list) == len(params)
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(len(var_list)):
                if 'depthwise' in var_list[i].name and len(params[i].shape) == 4:
                    params[i] = np.transpose(params[i], (0, 1, 3, 2))
                if len(params[i].shape) == 2:
                    params[i] = np.expand_dims(params[i], 0)
                    params[i] = np.expand_dims(params[i], 0)
                print(var_list[i].name, var_list[i].shape, params[i].shape)
                sess.run(tf.assign(var_list[i], params[i]))

            saver.save(sess, model_checkpoint_path, write_meta_graph=False,
                       write_state=False)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    init()
    acc_original = evaluate(model_checkpoint_path='log/model_dump/model.ckpt', has_bn=True,
                            qweight=False, qactivation=False, image_norm=True)
    fix_input_params()
    acc_fix_input = evaluate(model_checkpoint_path='log/model_dump/model_fix_input.ckpt', has_bn=True,
                             qweight=False, qactivation=False, image_norm=False)
    find_quantize_scale('log/model_dump/model_fix_input.ckpt')
    acc_int = evaluate(model_checkpoint_path='log/model_dump/model_fix_input.ckpt', has_bn=True,
                       qweight=True, qactivation=True, image_norm=False)
    bn_ema('log/model_dump/model_fix_input.ckpt', qweight=True, qactivation=True)
    acc_int_bn_ema = evaluate(model_checkpoint_path='log/model_dump/model_fix_input_bn_ema.ckpt', has_bn=True,
                              qweight=True, qactivation=True, image_norm=False)
    print('float acc = %.3f%%' % acc_original)
    print('float fix input = %.3f%%' % acc_fix_input)
    print('int acc = %.3f%%' % acc_int)
    print('int acc after bn ema = %.3f%%' % acc_int_bn_ema)
    train(model_checkpoint_path='log/model_dump/model_fix_input_bn_ema.ckpt',
          has_bn=True, qweight=True, qactivation=True, image_norm=False)


if __name__ == '__main__':
    main()
