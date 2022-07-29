# conding=utf-8
# # -*- coding:utf-8 -*-
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import glob
import os
import sys
import scipy.io as io
#import torch
from sklearn.model_selection import train_test_split

import provider
import tf_util
import modelnet_dataset

from tensorflow.python import debug as tf_debug


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # os.path.dirname(__file__)返回脚本的路径
ROOT_DIR = os.path.dirname(BASE_DIR)  # 语法：os.path.dirname(path) 功能：去掉文件名，返回目录   #print(os.path.dirname('W:\Python_File'))
# 结果
# W:\
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg_spec_cp',
                    help='Model name [default: pointnet2_cls_ssg_spec_cp]')
parser.add_argument('--subdir', default='', help='A sub dir that contains categoried models')
parser.add_argument('--log_dir', default='log11', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2000,help='Point Number [default: 1000]')  # ****************这个2000代表点数*** 改
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=10,help='Batch Size during training [default: 10]')  # ******************************* 改
parser.add_argument('--neighbor', type=int, default=32,help='neighbor during training [default: 2]')  # *******************************
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--debug', action='store_true', help='Whether to use debugger')
parser.add_argument('--eval_rotation', action='store_true', help='Whether to rotate in eval')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
neighbor = FLAGS.neighbor
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
# GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
SUBDIR = FLAGS.subdir
DEBUG = FLAGS.debug

sys.path.append(os.path.join(BASE_DIR, 'models', SUBDIR))

MODEL = importlib.import_module(
    FLAGS.model)  # import network module  #https://blog.csdn.net/xie_0723/article/details/78004649
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.subdir, FLAGS.model + '.py')

LOG_DIR_prefix = 'log'
EXP_prefix = ''
LOG_DIR = os.path.join(BASE_DIR, LOG_DIR_prefix, EXP_prefix, FLAGS.log_dir)
# LOG_DIR = 'F:\MLP\代码\classification\log\log11'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
# recording max eval accuracy and its model
LOG_FOUT_max_record = open(os.path.join(LOG_DIR, 'log_max_record.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 2  ################输入数据的类别#######################################

DATA_DIR = ROOT_DIR
DATA_PATH = os.path.join(DATA_DIR, 'data/modelnet10_normal_resampled')##########这个不用管


# def read_img(path):
#     cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
#     imgs = []
#     labels = []
#     for idx, folder in enumerate(cate):
#         for im in glob.glob(folder + '/*.mat'):
#             img = io.loadmat(im)
#             img = img['A']
#             imgs.append(img)
#             labels.append(idx)
#     return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
def read_img(path):
    cate=[path+'/'+x for x in os.listdir(path)]
    print(cate)
    imgs=[]
    labels=[]
    for idx,im in enumerate(cate):
        idx2=1
        img= io.loadmat(im)
        img=img['VERT']
        imgs.append(img)
        labels.append(idx2)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)


path = r'E:\MLP\mat\数据\mat格式\第三类'  ###############mat数据存放的路径
data, label = read_img(path)
# print(data,label)
# print(data.shape)
# print(data[2,:,:])
# train_data, test_data, train_labels, test_labels = train_test_split(data, label, train_size=0.4, test_size=0.6,
#                                                                     random_state=42)

train_data =data#[0:30,:,:]
test_data =data
train_labels =[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
test_labels =[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


print('train_data.shape', train_data.shape)
print('test_data.shape', test_data.shape)


TRAIN_DATASET = train_data
TEST_DATASET = test_data


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)  ######################不能注释，注释后不输出任何结果


def log_string_record(out_str, file):
    file.write(out_str + '\n')
    file.flush()
    print(out_str)  ######################不能注释，注释后不输出任何结果


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(  # tf.train.exponential_decay：指数衰减法
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():

        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

        is_training_pl = tf.placeholder(tf.bool, shape=())

        batch = tf.Variable(0) #初始化
        bn_decay = get_bn_decay(batch) #设置train的过程中学习率的衰减系数的
        tf.summary.scalar('bn_decay', bn_decay)
        print('333')


        ########################下面两行里面，包括整个页面里的所有l1_xyz,l1_points，都是你要输出的特征。如果想改动特征，就找有l1_xyz,l1_points的                          地方，增删改就行####################################
        pred, end_points,l1_xyz,l1_points= MODEL.get_model(pointclouds_pl,is_training_pl,bn_decay=bn_decay)
        l1_xyz,l1_points = huitu2(l1_xyz,l1_points)
        print('444')


        loss = MODEL.get_loss(pred, labels_pl, end_points)
        tf.summary.scalar('loss', loss)

        correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
        tf.summary.scalar('accuracy', accuracy)

        print("--- Get training operator")
        # Get training operator
        learning_rate = get_learning_rate(batch)
        tf.summary.scalar('learning_rate', learning_rate)
        if OPTIMIZER == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
        elif OPTIMIZER == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=batch)

        saver = tf.train.Saver()

        config = tf.ConfigProto()

        config.allow_soft_placement = True
        config.log_device_placement = False

        merged = tf.summary.merge_all()
        sess = tf.Session(config=config)

        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        init = tf.global_variables_initializer()

        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)


        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points,
               'l1_xyz': l1_xyz,
               'l1_points':l1_points}

        eval_acc_max_so_far = -1
        for epoch in range(MAX_EPOCH):  #（0~~~~9）
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_acc_epoch,l1_xyz,l1_points = eval_one_epoch(sess, ops, test_writer)
            print(l1_points.shape)

            # print(l1_points.shape)
            if epoch % 20 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

            if eval_acc_epoch > eval_acc_max_so_far:
                eval_acc_max_so_far = eval_acc_epoch
                log_string_record('**** EPOCH %03d ****' % (epoch), LOG_FOUT_max_record)
                log_string_record('eval accuracy: %f' % eval_acc_epoch, LOG_FOUT_max_record)
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_max_record.ckpt"))
                log_string_record("Model saved in file: %s" % save_path, LOG_FOUT_max_record)




            if epoch == 9:
                # for i in range(3):  BATCH_SIZE
                for k in range(len(TEST_DATASET)):
                    paras = {}
                    paras['model_mlp'] = l1_points[k:k + 1, :, :].reshape([2000,352])  #改
                    paras['mlp2'] = l1_xyz[k:k + 1, :, :].reshape([2000, 3])    #改
                    # paras['mlp'] = tf.squeeze(l1_points[k:k + 1, :, :])
                    if not os.path.isdir('./保存/'):
                        os.mkdir('./保存/')
                    io.savemat('./保存/' + '%.3d.mat' % k, paras)




def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET) // BATCH_SIZE

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        batch_data = get_batch(train_data, train_idxs, start_idx, end_idx)
        batch_label = get_batch0(train_labels, train_idxs, start_idx, end_idx)

#         aug_data = augment_batch_data(batch_data)
#         aug_data = provider.random_point_dropout(aug_data)

        aug_data = batch_data

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training, }

        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

    # if (batch_idx+1)%50 == 0:
    # log_string(' -- 88888 %03d / %03d 88888--' % (batch_idx + 1, num_batches))
    # ################训练精度多久输出一次跟输入数据量，还有这个数字有关系，如果输入数据太少，就会报错，  batch_idx
    # ###############报错显示：UnboundLocalError: local variable 'batch_idx' referenced before assignment#################################
    # log_string('mean loss: %f' % (loss_sum / 50))
    # log_string('accuracy: %f' % (total_correct / float(total_seen)))
    total_correct = 0
    total_seen = 0
    loss_sum = 0

def huitu2(l1_xyz,l1_points):

    l1_xyz = l1_xyz
    l1_points = l1_points
    return l1_xyz,l1_points


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    channel = dataset.shape[2]
    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, channel))
    for i in range(bsize):
        ps = dataset[idxs[i + start_idx]]
        batch_data[i] = ps
    return batch_data


def get_batch0(label, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    for i in range(bsize):
        cls = label[idxs[i + start_idx]]
        batch_label[i] = cls
    return batch_label


def eval_one_epoch(sess, ops, test_writer):
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = (len(TEST_DATASET) + BATCH_SIZE - 1) // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    # for batch_idx in range(1):
    aa_xyz = []
    aa_points = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(TEST_DATASET))
        # start_idx = 0                #改
        # end_idx = len(TEST_DATASET)  #改
        bsize = end_idx - start_idx
        batch_data = get_batch(test_data, test_idxs, start_idx, end_idx)
        batch_label = get_batch0(test_labels, test_idxs, start_idx, end_idx)

#         if FLAGS.eval_rotation:
#             aug_data = augment_batch_data(batch_data)
#         else:
#             aug_data = batch_data
        aug_data = batch_data

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}

        summary, step, loss_val, pred_val,l1_xyz,l1_points= sess.run(
            [ops['merged'], ops['step'],
             ops['loss'], ops['pred'], ops['l1_xyz'],ops['l1_points']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += (loss_val * float(bsize / BATCH_SIZE)) # 改
        # loss_sum += (loss_val * float(bsize / len(TEST_DATASET)))
        for i in range(start_idx, end_idx):
            l = batch_label[i - start_idx]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i - start_idx] == l)

        if batch_idx == 0:
            k = l1_xyz
            m = l1_points
        else:
            k = np.append(k, l1_xyz, axis=0)
            m = np.append(m, l1_points, axis=0)


    eval_acc = (total_correct / float(total_seen))
    eval_acc_class_avg = (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)))

    summary_acc = tf.Summary(value=[
        tf.Summary.Value(
            tag='eval_acc', simple_value=float(eval_acc)),
        tf.Summary.Value(
            tag='evl_acc_classavg', simple_value=float(eval_acc_class_avg))
    ])
    test_writer.add_summary(summary_acc, EPOCH_CNT)

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    EPOCH_CNT += 1
    #return total_correct / float(total_seen),l1_xyz,l1_points
    return total_correct / float(total_seen), k, m

def augment_batch_data(batch_data):

    if FLAGS.normal:
        rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
    else:
        rotated_data = provider.rotate_point_cloud(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)

    jittered_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
    jittered_data = provider.shift_point_cloud(jittered_data)
    jittered_data = provider.jitter_point_cloud(jittered_data)
    rotated_data[:, :, 0:3] = jittered_data
    return rotated_data


if __name__ == "__main__":
    #     log_string('pid: %s'%(str(os.getpid())))
    train()   #从这可以获得结果（需要输出保存文件的代码）  如果调用train可可以讲这部分去掉
    print('111')
    LOG_FOUT.close()

