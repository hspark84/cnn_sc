import os
import time
import tensorflow as tf
import numpy as np
from sys import argv
import ipdb


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from inputs import get_train_dataset_fb_id, get_eval_dataset_fb_id

# configurations
CLASSES = [
    'air conditioner',
    'car horn',
    'children playing',
    'dog bark',
    'drilling',
    'engine idling',
    'gun shot',
    'jackhammer',
    'siren',
    'street music'
]

train_set_cv6 = []
valid_set_cv6 = []
test_set_cv6 = []
for i in range(10):
  # test set
  test_idx = i
  test_set = [data_sets_all_6[i]]
  test_set_cv6.append(test_set)
  # valid set
  valid_idx = (i + 1) % 10
  valid_set = [data_sets_all_6[valid_idx]]
  valid_set_cv6.append(valid_set)
  # train set
  train_idx = range(10)
  train_idx = list(set(train_idx) - set([test_idx]) - set([valid_idx]))
  train_set = [data_sets_all_6[i] for i in train_idx]
  train_set_cv6.append(train_set)



os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.logging.set_verbosity(tf.logging.ERROR)


BATCH_SIZE = 100
EPOCHS = 50
LEARNING_RATE = 1e-4 
weight_decay = 1e-3
LR_DECAY_BASE = 1.00


IKP = 0.25
OKP = 0.50
SKP = 0.75

N_CLASSES = len(CLASSES)

CV_IDX = 0 # [0, ... 9]
trial = 0

tbpath = 'Tensorboard/mscnn_' + str(CV_IDX) + '_' + str(trial)
mpath = 'Model/mscnn_' + str(CV_IDX) + '_' + str(trial)


tf.gfile.MkDir(tbpath)
tf.gfile.MkDir(mpath)


def CNN_JS(X, is_training=False):
  # 1st convolutional layer, map spectrogram image to 24 feature maps
  h_conv1 = tf.layers.conv2d(X, 24, [5, 5], [1, 1], 'same', use_bias=False)
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='SAME') 
  h_bn1   = tf.layers.batch_normalization(h_pool1, training=is_training)
  h_1     = tf.nn.relu(h_bn1)
  # 2nd convolutional layer, map 24 feature maps to 48
  h_conv2 = tf.layers.conv2d(h_1, 48, [5, 5], [1, 1], 'same', use_bias=False)
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='SAME')
  h_bn2   = tf.layers.batch_normalization(h_pool2, training=is_training)
  h_2     = tf.nn.relu(h_bn2)
  # 3nd convolutional layer, map 48 feature maps to 48, no pooling
  h_conv3 = tf.layers.conv2d(h_2, 48, [5, 5], [1, 1], 'same', use_bias=False)
  h_bn3   = tf.layers.batch_normalization(h_conv3, training=is_training)
  h_3     = tf.nn.relu(h_bn3)
  return h_3


def FC1(X, nhid=100, kp=1.0, is_training=False):
  tsX = tf.shape(X)
  slX = X.get_shape().as_list()
  X = tf.reshape(X, [tsX[0], tsX[1], slX[2]*slX[3]])
  X = tf.reduce_mean(X, axis=1)		# temporal avg pooling
  X = tf.nn.dropout(X, keep_prob = kp)
  X = tf.layers.dense(X, nhid, use_bias=False)
  X = tf.layers.batch_normalization(X, training=is_training)
  X = tf.nn.relu(X)			# [B x 100]
  return X

def FC2(X, nhid=10, kp=1.0, is_training=False):
  X = tf.nn.dropout(X, keep_prob = kp)
  X = tf.layers.dense(X, nhid)
  return X


def Model_TF(X, LABELS_FB, lstm_ikp=1.0, lstm_okp=1.0, lstm_skp=1.0, is_training=False):

  # CNN feature map via shared CNN_JS
  cnn_feat = CNN_JS(X, is_training)
  # FC1 
  fc      = FC1(cnn_feat, FC1_DIM, lstm_ikp, is_training)
  # FC2 for sound classification
  fc2     = FC2(fc, N_CLASSES, lstm_okp, is_training)
  return fc2 

# Build our dataflow graph.
GRAPH = tf.Graph()
with GRAPH.as_default():

  # datasets
  dataset_train = get_train_dataset_fb_id(train_set_cv6[CV_IDX], batch_size=BATCH_SIZE)
  dataset_valid = get_eval_dataset_fb_id(valid_set_cv6[CV_IDX])
  dataset_test  = get_eval_dataset_fb_id(test_set_cv6[CV_IDX])

  is_training = tf.placeholder(tf.bool)

  # reinitializable iterator
  iterator        = tf.contrib.data.Iterator.from_structure(dataset_train.output_types, (tf.TensorShape([None, None, 128, 3]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]) ) )
  iter_init_train = iterator.make_initializer(dataset_train)
  iter_init_valid = iterator.make_initializer(dataset_valid)
  iter_init_test  = iterator.make_initializer(dataset_test)
  next_element = iterator.get_next()

  # spectrum normalization
  SPECTRUMS = next_element[0]
  LABELS    = next_element[1]
  LABELS_FB = next_element[2]

  # pass spectrum to model
  lstm_ikp = tf.placeholder(tf.float32, shape=())
  lstm_okp = tf.placeholder(tf.float32, shape=())
  lstm_skp = tf.placeholder(tf.float32, shape=())

  # Model
  pred_Y = Model_TF(SPECTRUMS, LABELS_FB, lstm_ikp, lstm_okp, lstm_skp, is_training)

  # sound classification loss 
  sc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_Y, labels=LABELS))
  tf.summary.scalar("sc_loss", sc_loss)
  
  # fianl cost with l2_loss
  tvars = tf.trainable_variables()
  l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if 'dense' in v.name and 'kernel' in v.name])
  tf.summary.scalar("l2_loss", l2_loss)
  COST = sc_loss + weight_decay*l2_loss

  # calc num of tvars 
  nvars = 0
  for var in tvars:
    sh = var.get_shape().as_list()
    print(var.name, sh)
    nvars += np.prod(sh)
  print(nvars, 'total variables')

  # Compute gradients.
  lr = tf.placeholder(tf.float32, shape=())
  OPTIMIZER = tf.train.AdamOptimizer(lr)	# default epsilon 1e-08
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    grads,_ = tf.clip_by_global_norm(tf.gradients(COST,tvars),1)		# compute gradients and do clipping
    APPLY_GRADIENT_OP = OPTIMIZER.apply_gradients(zip(grads, tvars))	# apply gradients

  # evaluation
  correct_pred = tf.equal(tf.argmax(pred_Y, 1), LABELS)
  ACCURACY = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

  # confusion matrix
  CONF_MAT = tf.confusion_matrix(LABELS, tf.argmax(pred_Y, 1), num_classes=10)

  SUMMARIES_OP = tf.summary.merge_all()


# Start training the model.
with tf.Session(graph=GRAPH) as SESSION:
  # initialize first
  SESSION.run(tf.global_variables_initializer())

  # Create a tensorflow summary writer.
  SUMMARY_WRITER = tf.summary.FileWriter(tbpath, graph=GRAPH)

  # Create a tensorflow graph writer.
  GRAPH_WRITER = tf.train.Saver(max_to_keep=EPOCHS)

  steps = 0 	# for tf.summary step
  best_cost = float('Inf')
  best_acc  = float(0)
  best_model = 0 
  for EPOCH in range(EPOCHS):
    # initialize an iterator over the training dataset.
    SESSION.run(iter_init_train)
    iters = 0
    costs = 0.0
    accs  = 0.0
    
    lr_decay = LR_DECAY_BASE ** (EPOCH)
    lr_epoch = LEARNING_RATE / lr_decay
    print('Epoch %d, Leanring rate = %.7f' % (EPOCH, lr_epoch))

    start_time = time.time()
    while True:
      try:
        _, summaries, COST_VAL, ACC_VAL = SESSION.run([APPLY_GRADIENT_OP, SUMMARIES_OP, COST, ACCURACY],
                            feed_dict={lstm_ikp: IKP, lstm_okp: OKP, lstm_skp: SKP, is_training: True, lr: lr_epoch})
        costs += COST_VAL
        accs  += ACC_VAL
        iters += 1
        if iters % 100 == 0:
          end_time = time.time()
          DURATION = end_time - start_time
          start_time = end_time
          print('Epoch %d, Iters %d, cost = %.6f (%.3f sec)' % (EPOCH, iters, (costs/iters), DURATION))
          SUMMARY_WRITER.add_summary(summaries, steps)
          steps += 1
      except tf.errors.OutOfRangeError:
        break
    GRAPH_WRITER.save(SESSION, mpath+'/model', global_step=EPOCH)	# save model after each epoch
    print('Epoch %d, Train cost = %.6f, acc = %.6f' % (EPOCH, (costs/iters), (accs/iters)))

    # validation after each epoch
    SESSION.run(iter_init_valid) 
    iters = 0 
    costs = 0.0
    accs  = 0.0
    while True:
      try:
        COST_VAL, ACC_VAL = SESSION.run([COST, ACCURACY], 
                            feed_dict={lstm_ikp: 1.0, lstm_okp: 1.0, lstm_skp: 1.0, is_training: False, lr: 0.0})
        costs += COST_VAL
        accs  += ACC_VAL
        iters += 1 
      except tf.errors.OutOfRangeError:
        break
    final_cost = (costs/iters)
    final_acc  = (accs/iters)
    print('Epoch %d, Valid cost = %.6f, acc = %.6f' % (EPOCH, final_cost, final_acc))
    if final_acc > best_acc:
      print('Best epoch is changed from %d to %d, Acc %.4f to %.4f' % (best_model, EPOCH, best_acc, final_acc))
      best_acc = final_acc
      best_model = EPOCH

    # test after each epoch
    SESSION.run(iter_init_test) 
    iters = 0 
    costs = 0.0
    accs  = 0.0
    while True:
      try:
        COST_VAL, ACC_VAL = SESSION.run([COST, ACCURACY], 
                            feed_dict={lstm_ikp: 1.0, lstm_okp: 1.0, lstm_skp: 1.0, is_training: False, lr: 0.0})
        costs += COST_VAL
        accs  += ACC_VAL
        iters += 1 
      except tf.errors.OutOfRangeError:
        break
    final_cost = (costs/iters)
    final_acc  = (accs/iters)
    print('Epoch %d, Test cost = %.6f, acc = %.6f' % (EPOCH, final_cost, final_acc))


  # test
  best_model_path = mpath + '/model-' + str(best_model)
  GRAPH_WRITER.restore(SESSION, best_model_path)
  print('Test using the best model %s' % best_model_path)
  SESSION.run(iter_init_test)
  iters = 0 
  accs = 0.0
  conf_mat = np.zeros((10,10)).astype(int)
  while True:
    try:
      [ACCS_VAL, CONF_MAT_VAL] = SESSION.run([ACCURACY, CONF_MAT], feed_dict={lstm_ikp: 1.0, lstm_okp: 1.0, lstm_skp: 1.0, is_training: False, lr: 0.0})
      accs += ACCS_VAL
      conf_mat = conf_mat + CONF_MAT_VAL
      iters += 1
    except tf.errors.OutOfRangeError:
      break
  print('Test Accuracy : %.4f' % (accs/iters))

  print(conf_mat)
  np.savetxt(best_model_path + '-confmat.txt', conf_mat)
  np.save(best_model_path + '-confmat.npy', conf_mat)

