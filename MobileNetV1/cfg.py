import os
import sys

sys.path.insert(0, '../lib')

# model config
bn_eps = 1e-3
bn_momentum = 0.99
weight_decay = 1e-5
first_conv_name = 'conv1'
w_bit = 4
a_bit = 4
qactivation_bn_ema = False
add_with_float = True
relu_max = 6.0
dtype = 'float32'

# data config
train_dir = '/home/lzy/DL_DATA/ILSVRC/Data/CLS-LOC/train/'
train_txt = '/home/lzy/dl_project/tensorflow-int-quantize/data/train.txt'
val_dir = '/home/lzy/DL_DATA/ILSVRC/Data/CLS-LOC/val/'
val_txt = '/home/lzy/dl_project/tensorflow-int-quantize/data/val.txt'

image_mean = [127.5, 127.5, 127.5]
image_std = [127.5, 127.5, 127.5]

min_size = 256
image_size = 224

batch_size = 128
data_num_threads = 20
eval_batch_size = 200
train_num = 1281127
val_num = 50000

# log config
model_path = 'log/model_dump'
log_path = 'log/tb_dump'
if not os.path.exists('log'):
    os.mkdir('log')
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(log_path):
    os.mkdir(log_path)

# training config
max_epoch = 30
timeline_log = False
init_lr = 0.001
log_step = 20
