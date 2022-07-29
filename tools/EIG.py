# EIG
#from skimage import io

import glob
import os
import tensorflow as tf
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from scipy.io import loadmat
#from scipy.sparse import diags
#import scipy
import scipy.sparse
import scipy.io as sio

# dis = './bed40_30o/'
#
# for i in range(23):
#
#     W = loadmat(dis + 'bed_%.3d.mat' % i)
#     W = W['D']
#
#     e_vals, e_vecs = np.linalg.eig(W)
#     print(e_vals.shape)
#     print(e_vecs.shape)
#     print(e_vecs,type(e_vecs))
#
#
#     params_to_save = {}
#     params_to_save['E'] = e_vecs
#     if not os.path.isdir('./bed40oE/'):
#         os.mkdir('./bed40oE/')
#
#     sio.savemat('./bed40oE/' + 'bed_%.3d.mat' % i, params_to_save)  # [7:10]   #改
# dis = './tf_artist_ms/'
dis = './第三类zuo_linyums_ms'
for i in range(20):

    W = loadmat(dis +'/' +'%.3d.mat' % i)
    # W = loadmat(dis + 'model_%d_dist.mat' % i)
    W = W['DD']

    e_vals, e_vecs = np.linalg.eig(W)
    print(e_vals.shape)
    print(e_vecs.shape)
    print(e_vecs,type(e_vecs))


    params_to_save = {}
    params_to_save['E'] = e_vecs
    # params_to_save['Z'] = e_vals
    if not os.path.isdir('./第三类zuo_linyums_msE/'):
        os.mkdir('./第三类zuo_linyums_msE/')

    sio.savemat('./第三类zuo_linyums_msE/' + '%.3d.mat' % i, params_to_save)  # [7:10]   #改
#     # sio.savemat('./tf_msE/' + 'model_%d.mat' % i, params_to_save)  # [7:10]   #改
'''
    batch_input = {'part_evecs':[[3], [3]], 'part_evecs_trans':[[3], [3]], 'part_shot':[[3], [3]], 'model_evecs':[[3], [3]], 'model_evecs_trans':[[3], [3]], 'model_shot':[[3], [3]]}
	batch_input['part_evecs'][0] = batch_input0['part_evecs'][0]
	batch_input['part_evecs'][1] = batch_input1['part_evecs'][1]

	batch_input['part_evecs_trans'][0] = batch_input0['part_evecs_trans'][0]
	batch_input['part_evecs_trans'][1] = batch_input1['part_evecs_trans'][1]

	batch_input['part_shot'][0] = batch_input0['part_shot'][0]
	batch_input['part_shot'][1] = batch_input1['part_shot'][1]

	batch_input['model_evecs'][0] = batch_input0['model_evecs'][0]
	batch_input['model_evecs'][1] = batch_input1['model_evecs'][1]

	batch_input['model_evecs_trans'][0] = batch_input0['model_evecs_trans'][0]
	batch_input['model_evecs_trans'][1] = batch_input1['model_evecs_trans'][1]

	batch_input['model_shot'][0] = batch_input0['model_shot'][0]
	batch_input['model_shot'][1] = batch_input1['model_shot'][1]

c = {'hh':[[0],[9]]}
c['hh'][0] = [[[1,2,3],[1,2,3],[1,2,3]],[4,5,6],[4,5,6],[4,5,6]]
c['hh'][1] = [[[1,2,3],[1,2,3],[1,2,3]],[4,5,6],[4,5,6],[4,5,6]]
print(c)'''



# # COS_jvli
# from skimage import io
# import glob
# import os
# import tensorflow as tf
# import numpy as np
# from scipy import io
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
#
# from scipy.io import loadmat
# #from scipy.sparse import diags
# #import scipy
# import scipy.sparse
# import scipy.io as sio

#W = loadmat("F:/__identity/activity/论文/data/D001.mat")


# def laplacian(W, normalized=True):
#     """Return the Laplacian of the weigth matrix."""
#
#     # Degree matrix.
#     d = W.sum(axis=0)
#     #print(d,type(d))
#     # Laplacian matrix.
#     if not normalized:
#         D = scipy.sparse.diags(d.A.squeeze(), 0)
#         L = D - W
#     else:
#         d += np.spacing(np.array(0, W.dtype))
#         #print(d,type(d))
#         d = 1 / np.sqrt(d)
#         #print(d,type(d))
#         d = d.A
#         #print((d,type(d)))
#         D = scipy.sparse.diags(d.squeeze(), 0)
#         #print(D,type(D))
#         I = scipy.sparse.identity(d.size, dtype=W.dtype)
#         #print(I,type(I))
#         print(I.shape)
#         print(D.shape)
#         print(W.shape)
#         W = tf.transpose(W, [0, 2, 1])
#         L = I - D * W * D
#         #print(L,type(L),L.shape)
#
#     # assert np.abs(L - L.T).mean() < 1e-9
#    # assert type(L) is scipy.sparse.csr.csr_matrix
#     return L
#
# dis = './air_30ms/'
# #W = loadmat("F:/数据集/模型1/2011分类/distance_bott/bottle_000.mat")    #改
# #d = sio.loadmat(FLAGS.dist_maps + 'airplane_%.3d.mat' % i_model)
# for i in range(13):
#     W = loadmat(dis + 'airplane_%.3d.mat' % i)
#     W = W['D']
#     W = np.mat(W)
# #W = scipy.sparse.csr_matrix(W)
#     #print(W,type(W))
#     L = laplacian(W)
#     print(L)
#
#     params_to_save = {}
#     params_to_save['L'] = L
#     if not os.path.isdir('./chair_L/'):
#         os.mkdir('./chair_L/')
#
#     sio.savemat('./chair_L/' + 'chair_%.3d.mat' % i, params_to_save)  # [7:10]   #改


# def laplacian(W, normalized=True):
#     """Return the Laplacian of the weigth matrix."""
#
#     # Degree matrix.
#     d = W.sum(axis=0)
#     #print(d,type(d))
#     # Laplacian matrix.
#     if not normalized:
#         D = scipy.sparse.diags(d.A.squeeze(), 0)
#         L = D - W
#     else:
#         d += np.spacing(np.array(0, W.dtype))
#         #print(d,type(d))
#         d = 1 / np.sqrt(d)
#         #print(d,type(d))
#         d = d.A
#         #print((d,type(d)))
#         D = scipy.sparse.diags(d.squeeze(), 0)
#         #print(D,type(D))
#         I = scipy.sparse.identity(d.size, dtype=W.dtype)
#         #print(I,type(I))
#         print(I.shape)
#         print(D.shape)
#         print(W.shape)
#         # W = tf.transpose(W, [1, 0])
#         L = I - D * W * D
#         #print(L,type(L),L.shape)
#
#     # assert np.abs(L - L.T).mean() < 1e-9
#    # assert type(L) is scipy.sparse.csr.csr_matrix
#     return L
#
# dis = './保存/'
# #W = loadmat("F:/数据集/模型1/2011分类/distance_bott/bottle_000.mat")    #改
# #d = sio.loadmat(FLAGS.dist_maps + 'airplane_%.3d.mat' % i_model)
# for i in range(20):
#     W = loadmat(dis + '%.3d.mat' % i)
#     W = W['jjtz']
#     W = np.mat(W)
# #W = scipy.sparse.csr_matrix(W)
#     #print(W,type(W))
#     L = laplacian(W)
#     print(L)
#
#     params_to_save = {}
#     params_to_save['L'] = L
#     if not os.path.isdir('./保存_L/'):
#         os.mkdir('./保存_L/')
#
#     sio.savemat('./保存_L/' + '%.3d.mat' % i, params_to_save)  # [7:10]   #改

# def get_adj_mat_cos(local_cord, flag_normalized=False, order=1):
#     in_shape = local_cord.get_shape().as_list()
#     #in_shape = shape(local_cord)
#     loc_matmul = tf.matmul(local_cord, local_cord, transpose_b=True, name='loc_matmul')
#     print(type(loc_matmul))
#
#     loc_norm = tf.norm(local_cord, axis=-1, keep_dims=True)
#     print(loc_norm)
#     loc_norm_matmul = tf.matmul(loc_norm, loc_norm, transpose_b=True, name='loc_norm_matmul')
#     print(loc_norm_matmul)
#     D = tf.divide(loc_matmul, loc_norm_matmul + 1e-8, name='cos_D')
#     print(D)
#
#     D = tf.exp(D*order)  # tf.exp(x, name=None) 计算e的次方
#     print(D)
#
#     if flag_normalized:
#         D_max = tf.reduce_max(tf.reshape(D, [in_shape[0], in_shape[1], in_shape[2] * in_shape[2]]), axis=-1)
#         D_max = tf.expand_dims(D_max, -1)
#         D_max = tf.expand_dims(D_max, -1)
#         D_max = tf.tile(D_max, [1, 1, in_shape[2], in_shape[2]])
#         D = tf.divide(D, D_max + 1e-8)
#         print(D)
#
#     adj_mat = D
#
#     return adj_mat
#
# dis = './第三类_30o_shot/'
# #W = loadmat("F:/数据集/模型1/2011分类/distance_bott/bottle_000.mat")    #改
# #d = sio.loadmat(FLAGS.dist_maps + 'airplane_%.3d.mat' % i_model)
# for i in range(20):   #0~27
#     #M = loadmat(dis + 'chair_%.3d.mat' % i)
#     M = loadmat(dis + '%.3d.mat' % i)
#     M = M['E']
#     #M = mat(M)
#     #M = scipy.sparse.csr_matrix(M)
#     #print(shape(M)[1])
#     #print(tf.shape(M))
#
#     M = tf.constant(M)
#
#     #print(W,type(W))
#     adj_mat = get_adj_mat_cos(M)
#
#     sess = tf.Session()
#     b = adj_mat.eval(session=sess)
#     print(b)
#     print(type(b))
#
#     #adj_mat = np.mat(adj_mat)
#     #print(adj_mat)
#     # params_to_save = {}
#     # params_to_save['D'] = adj_mat
#     if not os.path.isdir('./第三类_30oshot_cosd/'):
#         os.mkdir('./第三类_30oshot_cosd/')
#
#     #sio.savemat('./d_cos/' + 'chair_%.3d.mat' % i, params_to_save)  # [7:10]   #改 '%.3d.mat' % i
#
#     np.save(file=r'F:\数据集\模型1\2011分类\第三类_30oshot_cosd\cos_d.npy',arr=b)
#
#
#     mat = np.load(r'F:\数据集\模型1\2011分类\第三类_30oshot_cosd\cos_d.npy')
#     print(mat)
#     mat = mat[:,:]
#     # print(mat)
#     io.savemat(r'F:\数据集\模型1\2011分类\第三类_30oshot_cosd\%.3d.mat' % i,{'cos_d':mat})

# # mlp
#
# from skimage import io
# import glob
# import os
# import tensorflow as tf
# import numpy as np
# from scipy import io
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
# import torch
# from scipy.io import loadmat
# #from tensorflow import scope
# #from scipy.sparse import diags
# #import scipy
# import scipy.sparse
# import scipy.io as sio
#
#
# def conv1d(inputs,
#            num_output_channels,
#            kernel_size,
#            scope,
#            stride=[1],
#            padding='SAME',
#            use_xavier=True,
#            stddev=1e-3,
#            weight_decay=0.0
#            ):
#
#    with tf.variable_scope(scope) as sc:
#         num_in_channels = inputs.get_shape()[-1].value
#         kernel_shape = [kernel_size,
#                     num_in_channels, num_output_channels]
#         kernel = _variable_with_weight_decay('weights',
#                                             shape=kernel_shape,
#                                             use_xavier=use_xavier,
#                                             stddev=stddev,
#                                             wd=weight_decay)
#         outputs = tf.nn.conv1d(inputs, kernel,
#                             stride=stride,
#                             padding=padding)
#         # biases = _variable_on_cpu('biases', [num_output_channels],
#         #                         tf.constant_initializer(0.0))
#         # outputs = tf.nn.bias_add(outputs, biases)
#         #print(outputs.shape,outputs)
#
#         return outputs
#
#
# def _variable_on_cpu(name, shape, initializer, use_fp16=False):
#
#     dtype = tf.float16 if use_fp16 else tf.float32
#     var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
#     return var
#
# def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
#
#     if use_xavier:
#         initializer = tf.contrib.layers.xavier_initializer()
#     else:
#         initializer = tf.truncated_normal_initializer(stddev=stddev)
#     var = _variable_on_cpu(name, shape, initializer)
#     if wd is not None:
#         weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#         tf.add_to_collection('losses', weight_decay)
#     return var
# # sess = tf.Session()
# dis = './shi/'
# for i in range(1):   #0~27
#     M = loadmat(dis + 'chair_%.3d.mat' % i)
#     M = M['cos_jvli']
#     M = M.astype(np.float32)
#     M = tf.constant(M)
#     M = tf.expand_dims(M,0)
#
#     new_points = conv1d(M, 352, 1, padding='VALID', stride=[1],scope='aa%d'%(i) )
#     print(new_points)
#     torch.save(new_points,"./tensor_%.pth" % i)
#
# # a = tf.constant(2.1)  # 定义tensor常量
# # sess = tf.Session()
# #
# # io.savemat(r'E:\放假2021.7.19\模型1\2011分类\dcos\mlp_.mat',{'mlp':b})
# #
# # np.save(file=r'E:\放假2021.7.19\模型1\2011分类\dcos\mlp.npy',arr=b)
# #
# #
# #
# # params_to_save = {}
# # params_to_save['cos_jvli'] = new_points
# # if not os.path.isdir('./chaircos_/'):
# #     os.mkdir('./chaircos_/')
# #
# # sio.savemat('./chaircos_/' + 'chair_%.3d.mat' % i, params_to_save)

# import numpy as np
# from sklearn.manifold import Isomap
#
# data = np.load("samples_data.npy")
# isomap = Isomap(n_components=2, n_neighbors=5, path_method="auto")
# data_2d = isomap.fit_transform(X=data)
# geo_distance_metrix = isomap.dist_matrix_  # 测地距离矩阵，shape=[n_sample,n_sample]