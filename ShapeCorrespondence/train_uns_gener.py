import os
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio

from models_cg import fmnet_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dianshu import dianshu_

flags = tf.app.flags
FLAGS = flags.FLAGS

# training params
flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate.')  # OH: originally was 1e-3
flags.DEFINE_integer('batch_size', 1, 'batch size.')  # OH: originally was 32
flags.DEFINE_integer('queue_size', 10, '')

# architecture parameters
flags.DEFINE_integer('num_layers', 7, 'network depth')

flags.DEFINE_integer('num_evecs', 120,
                     'number of eigenvectors used for representation. The first 500 are precomputed and stored in input')  # 2002 453  353 1107 chai571

flags.DEFINE_integer('dim_shot', 352, '')
flags.DEFINE_integer('z_b', 3, '')
'''
flags.DEFINE_integer('num_vertices', 900,'')    #改了
'''
# data parameters
flags.DEFINE_string('models_dir', './data/network_chairms/',
                    '')  # 改了 data_第三类_new3  network_chair   network_bed40oE

flags.DEFINE_string('yuanshi_dir', './data/chai/',
                    '')

flags.DEFINE_string('dist_maps', './data/chair_ms/', '')  # 改了 matrix_第三类          distance_cha   bed40_30o

flags.DEFINE_string('lo', './Results/train_chair_cg2-8', 'directory to save models and results')  # 改了 第三类_new3
flags.DEFINE_integer('max_train_iter', 3000, '')
flags.DEFINE_integer('save_summaries_secs', 60, '')
flags.DEFINE_integer('save_model_secs', 1200, '')
flags.DEFINE_string('master', '', '')

# globals
error_vec_unsupervised = []
# error_vec_supervised = []
train_subjects = range(
    28)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]

flags.DEFINE_integer('num_poses_per_subject_total', 10, '')
dist_maps = {}
num_v = []


# def get_input_pair(batch_size=1, num_vertices=FLAGS.num_vertices):  #不让它等它不就行了么
def get_input_pair(batch_size=1, num_v_=num_v):
    dataset = 'train'
    '''
    batch_input = {'part_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
                   'model_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
                   'part_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices)),
                   'model_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices)),
                   'part_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot)),
                   'model_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot))
                   }


    batch_model_dist = np.zeros((batch_size, num_vertices, num_vertices))
    batch_part_dist = np.zeros((batch_size, num_vertices, num_vertices))
    '''
    # batch_part_ind2model_ind = np.zeros((batch_size,num_vertices))
    for i_batch in range(batch_size):

        i_model = np.random.choice(train_subjects)  # model #OH: randomize model subject index [0...7], for train -5
        i_part = np.random.choice(train_subjects)  # OH: randomize part subject index [0...7], for train            -9

        num_vertices1 = num_v_[i_model]
        num_vertices2 = num_v_[i_part]

        # for i in range(100):
        # 	if num_vertices2 == 15044 and num_vertices1 == 15044:
        # 		i_part = np.random.choice(train_subjects)
        # 		num_vertices1 = num_v_[i_model]
        # 		num_vertices2 = num_v_[i_part]
        # 	else:
        # 		break

        print(num_vertices1, num_vertices2)  # 1603 728
        if i_batch == 0:
            batch_input = {'part_evecs': np.zeros((batch_size, num_vertices2, FLAGS.num_evecs)),
                           'model_evecs': np.zeros((batch_size, num_vertices1, FLAGS.num_evecs)),
                           'part_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices2)),
                           'model_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices1)),
                           'part_shot': np.zeros((batch_size, num_vertices2, FLAGS.dim_shot)),
                           'model_shot': np.zeros((batch_size, num_vertices1, FLAGS.dim_shot)),
                           'part_vert': np.zeros((batch_size, num_vertices2, FLAGS.z_b)),
                           'model_vert': np.zeros((batch_size, num_vertices1, FLAGS.z_b)),
                           'part_valss': np.zeros((batch_size, FLAGS.num_evecs, FLAGS.num_evecs)),
                           'model_valss': np.zeros((batch_size, FLAGS.num_evecs, FLAGS.num_evecs)),
                           }


            batch_model_dist = np.zeros((batch_size, num_vertices1, num_vertices1))
            batch_part_dist = np.zeros((batch_size, num_vertices2, num_vertices2))

            batch_input_, batch_model_dist_, batch_part_dist_ = get_pair_from_ram(i_model, i_part, dataset)
            # input_data('model_shot'---'part_shot',  'model_vert'---'part_vert')      m_star--p_star 直接是数了

            model_vals_, model_evecs_ = np.linalg.eig(batch_model_dist_)
            part_vals_, part_evecs_ = np.linalg.eig(batch_part_dist_)

            part_vals = part_vals_[0:571].reshape([571])
            model_vals = model_vals_[0:571].reshape([571])

            part_val = np.diag(part_vals)
            model_val = np.diag(model_vals)  #可以传了 特征值对角矩阵 （120,120）

            part_evec = part_evecs_[:, 0:571]
            model_evec = model_evecs_[:, 0:571]  #可以传了 120特征向量  （n，120）

            part_evecs_trans = part_evec.transpose([1, 0])
            model_evecs_trans = model_evec.transpose([1, 0]) #可以传了 特征向量转置 （120， n）

            batch_model_dist[i_batch] = batch_model_dist_  # slice the subsampled indices
            batch_part_dist[i_batch] = batch_part_dist_

            batch_input['part_evecs'][i_batch] = part_evec
            batch_input['part_evecs_trans'][i_batch] = part_evecs_trans
            batch_input['part_shot'][i_batch] = batch_input_['part_shot']
            batch_input['model_evecs'][i_batch] = model_evec
            batch_input['model_evecs_trans'][i_batch] = model_evecs_trans
            batch_input['model_shot'][i_batch] = batch_input_['model_shot']
            batch_input['part_vert'][i_batch] = batch_input_['part_vert']
            batch_input['model_vert'][i_batch] = batch_input_['model_vert']
            batch_input['part_valss'][i_batch] = part_val
            batch_input['model_valss'][i_batch] = model_val

        if i_batch == 1:
            kkk = 1

    return batch_input, batch_model_dist, batch_part_dist  # , batch_part_ind2model_ind


def get_pair_from_ram(i_model, i_part, dataset):  # 5 9 train
    input_data = {}  # （'part_evecs'，'part_evecs_trans'，'part_shot'，'model_s','model_evecs','model_evecs_trans','model_shot','shot_params',）8个

    if dataset == 'train':
        input_data['part_vert'] = models_train[i_part]['model_vert']
        input_data['part_shot'] = models_train[i_part]['model_shot']

        # input_data.update(models_train[i_model])
        input_data['model_vert'] = models_train[i_model]['model_vert']
        input_data['model_shot'] = models_train[i_model]['model_shot']

    # m_star from dist_map
    # m_star = dist_maps[i_subject_model]
    # p_star = dist_maps[i_subject_part]
    d = sio.loadmat(FLAGS.dist_maps + 'chair_%.3d.mat' % i_model)
    m_star = d['D']

    d = sio.loadmat(FLAGS.dist_maps + 'chair_%.3d.mat' % i_part)
    p_star = d['D']

    return input_data, m_star, p_star  #input_data('model_shot'---'part_shot',  'model_vert'---'part_vert')      m_star--p_star 直接是数了


def load_models_to_ram():
    global models_train
    models_train = {}

    # load model and part
    # for i_subject in train_subjects:
    for i_model in range(28):
        model_file = FLAGS.models_dir + 'chair_%.3d.mat' % i_model
        yuan_file = FLAGS.yuanshi_dir + 'chair_%.3d.mat' % i_model
        inputdata1 = sio.loadmat(model_file)
        inputdata2 = sio.loadmat(yuan_file)
        input_data = {}
        input_data['model_shot'] = inputdata1['model_shot']
        input_data['model_vert'] = inputdata2['VERT']
        # input_data['model_evecs'] = input_data['model_evecs'][:, 0:FLAGS.num_evecs]
        # input_data['model_evecs_trans'] = input_data['model_evecs_trans'][0:FLAGS.num_evecs, :]
        models_train[i_model] = input_data  # model_train[0~10]---000-shot_params,model_evecs,model_s,model_shot,model_evecs_trans-5个


def load_dist_maps():
    print('loading dist maps...')
    # load distance maps to memory for training set
    for i_subject in train_subjects:
        global dist_maps
        d = sio.loadmat(FLAGS.dist_maps + 'tr_reg_%.3d.mat' % (i_subject * FLAGS.num_poses_per_subject_total))
        dist_maps[i_subject] = d['D']


def run_training():
    print('lo=%s' % FLAGS.lo)
    if not os.path.isdir(FLAGS.lo):
        os.makedirs(FLAGS.lo)  # changed from mkdir

    print('num_evecs=%d' % FLAGS.num_evecs)

    print('building graph...')
    print('bad')
    with tf.Graph().as_default():

        # Set placeholders for inputs                    如果 想放特征值 现在这写！！！
        part_shot = tf.placeholder(tf.float32, shape=(None, None, FLAGS.dim_shot), name='part_shot')  # 都写成none不行么
        model_shot = tf.placeholder(tf.float32, shape=(None, None, FLAGS.dim_shot), name='model_shot')
        model_dist_map = tf.placeholder(tf.float32, shape=(None, None, None), name='model_dist_map')
        part_dist_map = tf.placeholder(tf.float32, shape=(None, None, None), name='part_dist_map')
        part2model_ind_gt = tf.placeholder(tf.float32, shape=(None, None), name='part2model_groundtruth')
        part_evecs = tf.placeholder(tf.float32, shape=(None, None, FLAGS.num_evecs), name='part_evecs')
        part_evecs_trans = tf.placeholder(tf.float32, shape=(None, FLAGS.num_evecs, None), name='part_evecs_trans')
        model_evecs = tf.placeholder(tf.float32, shape=(None, None, FLAGS.num_evecs), name='model_evecs')
        model_evecs_trans = tf.placeholder(tf.float32, shape=(None, FLAGS.num_evecs, None), name='model_evecs_trans')

        part_vert = tf.placeholder(tf.float32, shape=(None, None, FLAGS.z_b), name='part_vert')
        model_vert = tf.placeholder(tf.float32, shape=(None, None, FLAGS.z_b), name='model_vert')
        part_valss = tf.placeholder(tf.float32, shape=(None, FLAGS.num_evecs, FLAGS.num_evecs), name='part_valss')
        model_valss = tf.placeholder(tf.float32, shape=(None, FLAGS.num_evecs, FLAGS.num_evecs), name='model_valss')

        # train\test switch flag
        phase = tf.placeholder(dtype=tf.bool, name='phase')

        # net_loss, unsupervised_loss, safeguard_inverse, merged, P_norm, net = fmnet_model(phase, part_shot, model_shot,
        #  part_dist_map , model_dist_map, part2model_ind_gt,
        #  part_evecs, part_evecs_trans, model_evecs, model_evecs_trans)

        unsupervised_loss, safeguard_inverse, merged, P_norm, net = fmnet_model(phase, part_shot, model_shot, part_dist_map,
                                                                                model_dist_map, part_vert, model_vert, part_evecs, part_evecs_trans,
                                                                                model_evecs, model_evecs_trans, part_valss, model_valss
                                                                                )

        summary = tf.summary.scalar("num_evecs", float(FLAGS.num_evecs))

        global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        train_op = optimizer.minimize(unsupervised_loss, global_step=global_step)  # 少一个    aggregation_method=2

        saver = tf.train.Saver(max_to_keep=100)  # 保存模型和载入模型 用的
        sv = tf.train.Supervisor(logdir=FLAGS.lo,
                                 init_op=tf.global_variables_initializer(),
                                 local_init_op=tf.local_variables_initializer(),
                                 global_step=global_step,
                                 save_summaries_secs=FLAGS.save_summaries_secs,
                                 save_model_secs=FLAGS.save_model_secs,
                                 summary_op=None,
                                 saver=saver)

        writer = sv.summary_writer

        print('starting session...')
        iteration = 0
        with sv.managed_session(master=FLAGS.master) as sess:
            print('loading data to ram...')
            load_models_to_ram()
            num_v = dianshu_()
            print(num_v)
            # model_train[0~10]---000-shot_params,model_evecs,model_s,model_shot,model_evecs_trans-5个

            # This command loads the distance matrices to RAM.
            # If you can load all the distance matrices to RAM do it!
            # Unless, if it is too heavy, read each time the corresponding distance matrix from the hard disk
            # load_dist_maps()

            print('starting training loop...')
            while not sv.should_stop() and iteration < FLAGS.max_train_iter:
                iteration += 1
                start_time = time.time()

                # input_data, mstar, pstar, p2m_ind_gt = get_input_pair(FLAGS.batch_size)
                input_data, mstar, pstar = get_input_pair(FLAGS.batch_size, num_v)


                feed_dict = {phase: True,
                             part_shot: input_data['part_shot'],
                             model_shot: input_data['model_shot'],
                             model_dist_map: mstar,
                             part_dist_map: pstar,
                             # part2model_ind_gt: p2m_ind_gt,
                             part_evecs: input_data['part_evecs'],
                             part_evecs_trans: input_data['part_evecs_trans'],
                             model_evecs: input_data['model_evecs'],
                             model_evecs_trans: input_data['model_evecs_trans'],
                             part_vert: input_data['part_vert'],
                             model_vert: input_data['model_vert'],
                             part_valss: input_data['part_valss'],
                             model_valss: input_data['model_valss']

                             }

                # summaries, step, my_loss, my_unsupervised_loss, safeguard, _ = sess.run(
                # [merged, global_step, net_loss, unsupervised_loss, safeguard_inverse, train_op], feed_dict=feed_dict)
                summaries, step, my_unsupervised_loss, safeguard, _ = sess.run(
                    [merged, global_step, unsupervised_loss, safeguard_inverse, train_op], feed_dict=feed_dict)

                writer.add_summary(summaries, step)
                summary_ = sess.run(summary)
                writer.add_summary(summary_, step)

                duration = time.time() - start_time

                # print('train - step %d: loss = %.4f unsupervised loss = %.4f(%.3f sec)' % (step, my_loss, my_unsupervised_loss, duration))
                print('train - step %d: unsupervised loss = %.4f(%.3f sec)' % (step, my_unsupervised_loss, duration))

                error_vec_unsupervised.append(my_unsupervised_loss)
            # error_vec_supervised.append(my_loss)

            saver.save(sess, FLAGS.lo + '/model_unsupervised.ckpt', global_step=step)
            writer.flush()
            sv.request_stop()
            sv.stop()

    # OH: save training error
    params_to_save = {}
    params_to_save['error_vec_unsupervised'] = np.array(error_vec_unsupervised)
    # params_to_save['error_vec_supervised'] = np.array(error_vec_supervised)
    sio.savemat(FLAGS.lo + '/training_error.mat', params_to_save)  # log_dir

    # OH: plot training error
    hu = plt.plot(np.array(error_vec_unsupervised), 'r')
    # hs = plt.plot(np.array(error_vec_supervised),'b')
    red_patch = mpatches.Patch(color='red', label='Unsupervised')
    # blue_patch = mpatches.Patch(color='blue', label='Supervised')
    # plt.legend(handles=[red_patch,blue_patch])
    plt.legend(handles=[red_patch])
    plt.title('Training process with the unsupervised loss')
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.show()


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
