from datetime import datetime
import os.path
import os
import time
import math
import tensorflow as tf
from tensorflow.python.client import timeline
import pickle
from tqdm import tqdm

import cfg
import net
import dataset
from utils import get_learning_rate
import layer


def evaluate(model_checkpoint_path='log/model_dump/model.ckpt', has_bn=True,
             qweight=False, qactivation=False, image_norm=False):
    scale = None
    if qweight or qactivation:
        with open('log/scale', 'rb') as f:
            scale = pickle.load(f)

    graph = tf.Graph()
    with graph.as_default():
        iterator = dataset.make_val_dataset()
        images, labels = iterator.get_next()
        val_logits = net.inference(images, False, has_bn=has_bn, image_norm=image_norm,
                               qweight=qweight, qactivation=qactivation, scale=scale)
        val_acc = 100 * tf.reduce_sum(tf.cast(tf.nn.in_top_k(val_logits, labels, 1), dtype=tf.float32))

        var_list = tf.get_collection('params')
        saver = tf.train.Saver(var_list)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_checkpoint_path)

            eval_acc = 0
            num_epoch = int(math.ceil(cfg.val_num / cfg.eval_batch_size))
            # num_epoch = 10
            for _ in tqdm(range(num_epoch)):
                _val_acc = sess.run(val_acc)
                eval_acc += _val_acc
            return eval_acc / cfg.val_num


def train(model_checkpoint_path = 'log/model_dump/model_fix_input_bn_ema.ckpt',
          has_bn=True, qweight=False, qactivation=False, image_norm=False):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(dtype=tf.float32)

        with tf.device('/cpu:0'):
            train_iter = dataset.make_train_dataset()
            image_batch, label_batch = train_iter.get_next()

        print('images:', image_batch.shape, image_batch.dtype)
        # Build inference Graph.
        scale = None
        if qweight or qactivation:
            with open('log/scale', 'rb') as f:
                scale = pickle.load(f)
        logits = net.inference(image_batch, phase_train=True, has_bn=has_bn, image_norm=image_norm,
                               qactivation=qactivation, qweight=qweight, scale=scale)

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        total_loss, softmax_loss, acc = layer.loss(logits, label_batch)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('softmax_loss', softmax_loss)
        tf.summary.scalar('acc', acc)

        train_op = layer.train(total_loss, learning_rate, global_step)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(cfg.log_path, tf.get_default_graph())

        pre_saver = tf.train.Saver(tf.get_collection('params'))

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=5000)

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=False,
                                                gpu_options=tf.GPUOptions(allow_growth=True)))
        sess.run(init)

        if model_checkpoint_path is not None:
            pre_saver.restore(sess, model_checkpoint_path)
            print('init model from {}'.format(model_checkpoint_path))

        best_val_acc = evaluate(model_checkpoint_path, has_bn=True,
                                qweight=True, qactivation=True, image_norm=False)
        if cfg.timeline_log:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            options = None
            run_metadata = None
        start_time = time.time()

        train_num_batch = cfg.train_num // cfg.batch_size
        train_log = open('log/work_log.txt', 'w')
        for epoch in range(cfg.max_epoch):
            for step in range(0, train_num_batch):
                lr = get_learning_rate(epoch, step, train_num_batch)
                feed_dict = {learning_rate: lr}
                if step % cfg.log_step == 0:
                    _, _total_loss, _softmax_loss, _acc, _summary = \
                        sess.run([train_op, total_loss, softmax_loss, acc, summary_op],
                                 feed_dict=feed_dict,
                                 options=options,
                                 run_metadata=run_metadata
                                 )
                    duration = float(time.time() - start_time) / cfg.log_step
                    examples_per_sec = cfg.batch_size / duration
                    log_line = "%s: Epoch=%d/%d, Step=%d/%d, lr=%.7f, total_loss=%.3f, softmax_loss=%.3f, " \
                               "acc=%.2f%%(%.1f examples/sec; %.3f sec/batch)" \
                               % (datetime.now().strftime('%m-%d %H:%M:%S'), epoch, cfg.max_epoch,
                                  step, train_num_batch, lr,
                                  _total_loss, _softmax_loss, _acc,
                                  examples_per_sec, duration)
                    train_log.write(log_line + '\n')
                    print(log_line)
                    summary_writer.add_summary(_summary, global_step=step)
                    start_time = time.time()

                    if cfg.timeline_log:
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        with open('timeline.json', 'w') as wd:
                            wd.write(ctf)

                else:
                    _ = sess.run(train_op, feed_dict=feed_dict)

            saver.save(sess, '{}/model.ckpt'.format(cfg.model_path), global_step=epoch)
            _val_acc = evaluate('{}/model.ckpt-{}'.format(cfg.model_path, epoch), has_bn=True,
                                qweight=True, qactivation=True, image_norm=False)

            if _val_acc > best_val_acc:
                pre_saver.save(sess, '{}/best_model.ckpt'.format(cfg.model_path), write_meta_graph=False,
                               write_state=False, global_step=None)
                best_val_acc = _val_acc

            val_log = 'epoch=%d, val_acc=%.3f%%, best_val_acc=%.3f%%' \
                      % (epoch, _val_acc, best_val_acc)
            train_log.write(val_log + '\n')
            print(val_log)
