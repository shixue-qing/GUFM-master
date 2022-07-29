import os
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio

from models_cg import fmnet_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

flags = tf.app.flags
FLAGS = flags.FLAGS

# training params
flags.DEFINE_float('learning_rate', 1e-3, 'initial learning rate.')
flags.DEFINE_integer('batch_size', 2, 'batch size.')
flags.DEFINE_integer('queue_size', 10, '')

# architecture parameters
flags.DEFINE_integer('num_layers', 7, 'network depth')
flags.DEFINE_integer('num_evecs', 120,
                     'number of eigenvectors used for representation. The first 500 are precomputed and stored in input')
flags.DEFINE_integer('dim_shot', 352, '')
flags.DEFINE_integer('num_vertices', 2000, '')
flags.DEFINE_integer('z_b', 3, '')
# data parameters
flags.DEFINE_string('models_dir', './data/network_shrec_ms120/', '')

flags.DEFINE_string('yuanshi_dir', './data/shrec/',
                    '')

flags.DEFINE_string('dist_maps', './data/shrec_ms/', '')

flags.DEFINE_string('lo', './Results/train_shrecms120cg', 'directory to save models and results')
flags.DEFINE_integer('max_train_iter', 3000, '')
flags.DEFINE_integer('save_summaries_secs', 60, '')
flags.DEFINE_integer('save_model_secs', 1200, '')
flags.DEFINE_string('master', '', '')

# globals
error_vec_unsupervised = []
# train_subjects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
train_subjects = range(20)
flags.DEFINE_integer('num_poses_per_subject_total', 10, '')
dist_maps = {}


def get_input_pair(batch_size=1, num_vertices=FLAGS.num_vertices):
    dataset = 'train'
    batch_input = {'part_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
                   'model_evecs': np.zeros((batch_size, num_vertices, FLAGS.num_evecs)),
                   'part_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices)),
                   'model_evecs_trans': np.zeros((batch_size, FLAGS.num_evecs, num_vertices)),
                   'part_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot)),
                   'model_shot': np.zeros((batch_size, num_vertices, FLAGS.dim_shot)),

                   'part_vert': np.zeros((batch_size, num_vertices, FLAGS.z_b)),
                   'model_vert': np.zeros((batch_size, num_vertices, FLAGS.z_b)),
                   'part_valss': np.zeros((batch_size, FLAGS.num_evecs, FLAGS.num_evecs)),
                   'model_valss': np.zeros((batch_size, FLAGS.num_evecs, FLAGS.num_evecs))
                   }

    batch_model_dist = np.zeros((batch_size, num_vertices, num_vertices))
    batch_part_dist = np.zeros((batch_size, num_vertices, num_vertices))
    for i_batch in range(batch_size):
        i_model = np.random.choice(train_subjects)  # model #OH: randomize model subject index [0...7], for train -5
        i_part = np.random.choice(train_subjects) #OH: randomize part subject index [0...7], for train            -9
        # i_part = np.int32(10)

        batch_input_, batch_model_dist_, batch_part_dist_ = get_pair_from_ram(i_model, i_part, dataset)

        model_vals_, model_evecs_ = np.linalg.eig(batch_model_dist_)
        part_vals_, part_evecs_ = np.linalg.eig(batch_part_dist_)

        part_vals = part_vals_[0:FLAGS.num_evecs].reshape([FLAGS.num_evecs])
        model_vals = model_vals_[0:FLAGS.num_evecs].reshape([FLAGS.num_evecs])

        part_val = np.diag(part_vals)
        model_val = np.diag(model_vals)  # （120,120）

        part_evec = part_evecs_[:, 0:FLAGS.num_evecs]
        model_evec = model_evecs_[:, 0:FLAGS.num_evecs]  #  （n，120）

        part_evecs_trans = part_evec.transpose([1, 0])
        model_evecs_trans = model_evec.transpose([1, 0])  # （120， n）

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

    return batch_input, batch_model_dist, batch_part_dist


def get_pair_from_ram(i_model, i_part, dataset):
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
    d = sio.loadmat(FLAGS.dist_maps + '%.3d.mat' % i_model)
    m_star = d['D']

    d = sio.loadmat(FLAGS.dist_maps + '%.3d.mat' % i_part)
    p_star = d['D']

    return input_data, m_star, p_star


def jiazai():
	m_ = r'./save/'
	n_ = r'./data/network_shrec_ms120/'
	for i_num in range(20):
		m_m = m_ + '%.3d.mat' % i_num
		n_n = n_ + '%.3d.mat' % i_num
		m_m_m = sio.loadmat(m_m)
		n_n_n = sio.loadmat(n_n)
		del n_n_n['model_shot']
		n_n_n['model_shot'] = m_m_m['model_mlp']
		# print(n_n_n)
		para_s = {}
		para_s['model_evecs'] = n_n_n['model_evecs']
		para_s['model_evecs_trans'] = n_n_n['model_evecs_trans']
		para_s['model_shot'] = n_n_n['model_shot']
		para_s['shot_params'] = n_n_n['shot_params']
		#para_s['model_shot'] =n_n_n['model_shot']
		if not os.path.isdir('./data/new_mlp352/'):
			os.mkdir('./data/new_mlp352/')
		sio.savemat('./data/new_mlp352/' + '%.3d.mat' % i_num, para_s)


def load_models_to_ram():
    global models_train
    models_train = {}
    models_dir_ = r'./data/new_mlp352/'

    # load model and part
    # for i_subject in train_subjects:
    for i_model in range(20):
        model_file = models_dir_ + '%.3d.mat' % i_model
        yuan_file = FLAGS.yuanshi_dir + '%.3d.mat' % i_model
        inputdata1 = sio.loadmat(model_file)
        inputdata2 = sio.loadmat(yuan_file)
        input_data = {}
        input_data['model_shot'] = inputdata1['model_shot']
        input_data['model_vert'] = inputdata2['VERT']
        # input_data['model_evecs'] = input_data['model_evecs'][:, 0:FLAGS.num_evecs]
        # input_data['model_evecs_trans'] = input_data['model_evecs_trans'][0:FLAGS.num_evecs, :]
        models_train[
            i_model] = input_data  # model_train[0~10]---000-shot_params,model_evecs,model_s,model_shot,model_evecs_trans-5个





def run_training():
    print('lo=%s' % FLAGS.lo)
    if not os.path.isdir(FLAGS.lo):
        os.makedirs(FLAGS.lo)  # changed from mkdir
    print('num_evecs=%d' % FLAGS.num_evecs)
    print('building graph...')
    with tf.Graph().as_default():

        # Set placeholders for inputs
        part_shot = tf.placeholder(tf.float32, shape=(None, None, FLAGS.dim_shot), name='part_shot')
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

        unsupervised_loss, safeguard_inverse, merged, P_norm, net = fmnet_model(phase, part_shot, model_shot, part_dist_map,
                                                                                model_dist_map, part_vert, model_vert, part_evecs, part_evecs_trans,
                                                                                model_evecs, model_evecs_trans, part_valss, model_valss)
        summary = tf.summary.scalar("num_evecs", float(FLAGS.num_evecs))

        global_step = tf.Variable(0, name='global_step', trainable=False)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        train_op = optimizer.minimize(unsupervised_loss, global_step=global_step)

        saver = tf.train.Saver(max_to_keep=100)
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
            jiazai()
            load_models_to_ram()

            # This command loads the distance matrices to RAM.
            # If you can load all the distance matrices to RAM do it!
            # Unless, if it is too heavy, read each time the corresponding distance matrix from the hard disk
            # load_dist_maps()

            print('starting training loop...')
            while not sv.should_stop() and iteration < FLAGS.max_train_iter:
                iteration += 1
                start_time = time.time()

                input_data, mstar, pstar = get_input_pair(FLAGS.batch_size)

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
                             model_valss: input_data['model_valss'],
                             }

                summaries, step, my_unsupervised_loss, safeguard, _ = sess.run(
                    [merged, global_step, unsupervised_loss, safeguard_inverse, train_op],
                    feed_dict=feed_dict)

                writer.add_summary(summaries, step)
                summary_ = sess.run(summary)
                writer.add_summary(summary_, step)

                duration = time.time() - start_time

                print('train - step %d: unsupervised loss = %.4f(%.3f sec)' % (
                step, my_unsupervised_loss, duration))
                error_vec_unsupervised.append(my_unsupervised_loss)

            saver.save(sess, FLAGS.lo + '/model_unsupervised.ckpt', global_step=step)
            writer.flush()
            sv.request_stop()
            sv.stop()

    # OH: save training error
    params_to_save = {}
    params_to_save['error_vec_unsupervised'] = np.array(error_vec_unsupervised)
    sio.savemat(FLAGS.lo + '/training_error.mat', params_to_save)  # log_dir

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


import sys
sys.path.append(r'E:\MLP\代码\classification')
import train读取modelnet2
def main(_):
	train读取modelnet2.train()
	run_training()


if __name__ == '__main__':
	tf.app.run()
