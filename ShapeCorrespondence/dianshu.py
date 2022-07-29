import os
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio

from models import fmnet_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
flags = tf.app.flags
FLAGS = flags.FLAGS










def dianshu_():
    models_dir = './faust_synthetic/chai/'
    models_train = {}
    num_v = []

    for i_model in range(28):
	    model_file = models_dir + 'chair_%.3d.mat' % i_model
	    input_data = sio.loadmat(model_file)
	    models_train[i_model] = input_data  #model_train[0~10]---000-shot_params,model_evecs,model_s,model_shot,model_evecs_trans-5个

    for i in range(28):
        tt = models_train[i]['n'][0,0]
        num_v.append(tt)

    return num_v