import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module  
from pointnet_util import pointnet_sa_module_spec  
from scipy.io import savemat

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
  
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    end_points['l0_xyz'] = l0_xyz

    ##############这个是网络里的三层mlp输出############################
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=128, radius=0.4, nsample=32, mlp=[64,64,352], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    
    
    net = tf.reshape(l1_points, [batch_size, -1])
    

    net = tf_util.fully_connected(net,512, activation_fn=None, scope='fc4')
    ####################这个2代表类别数######################
    net = tf_util.fully_connected(net,2, activation_fn=None, scope='fc5')
  
    
    return net,end_points,l1_xyz, l1_points
    

def get_loss(pred, label, end_points):

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
