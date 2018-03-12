import tensorflow as tf
import ipdb

def read_and_decode(filename_queue, specgram_shape):

  reader = tf.TFRecordReader()
  # Read an example from the TFRecords file.
  _, example = reader.read(filename_queue)
  features = tf.parse_single_example(example, features={
    'spectrum': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64)
  })
  spectrum = tf.decode_raw(features['spectrum'], tf.float32)
  spectrum.set_shape(specgram_shape)
  label = tf.cast(features['label'], tf.int64)
  return spectrum, label

# for multi-scale input (T x F x C spectrums)
def _parse_function_train_fb_id(example_proto):
  # parsing
  features = {'spectrum': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64),
              'label_fb': tf.FixedLenFeature([], tf.int64),
              'wav_id': tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)

  # spectrum, repeating or random cropping
  spectrum = tf.decode_raw(parsed_features['spectrum'], tf.float32)
  spectrum = tf.reshape(spectrum, [-1, 128, 3])

  # repeating
  spectrum = tf.cond(tf.less(tf.shape(spectrum)[0], 128),
                     lambda: tf.tile(spectrum, [tf.floordiv(128,tf.shape(spectrum)[0])+1, 1, 1]),
                     lambda: spectrum)

  # patching
  spectrum = tf.random_crop(spectrum, [128, 128, 3])

  # label
  label    = tf.cast(parsed_features['label'], tf.int64)

  # label_fb
  label_fb = tf.cast(parsed_features['label_fb'], tf.int64)

  # waf_id
  wav_id = parsed_features['wav_id']

  return spectrum, label, label_fb, wav_id

def _parse_function_eval_fb_id(example_proto):
 # parsing
  features = {'spectrum': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64),
              'label_fb': tf.FixedLenFeature([], tf.int64),
              'wav_id': tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)

  # spectrum, repeating or random cropping
  spectrum = tf.decode_raw(parsed_features['spectrum'], tf.float32)
  spectrum = tf.reshape(spectrum, [-1, 128, 3])

  # repeating
  spectrum = tf.cond(tf.less(tf.shape(spectrum)[0], 128),
                     lambda: tf.slice(tf.tile(spectrum, [tf.floordiv(128,tf.shape(spectrum)[0])+1, 1, 1]), [0, 0, 0], [128, 128, 3]), lambda: spectrum)

  # label
  label    = tf.cast(parsed_features['label'], tf.int64)

  # label_fb
  label_fb = tf.cast(parsed_features['label_fb'], tf.int64)

  # waf_id
  wav_id = parsed_features['wav_id']
  
  return spectrum, label, label_fb, wav_id


def get_train_dataset_fb_id(file_paths, batch_size=100):
  dataset = tf.contrib.data.TFRecordDataset(file_paths)
  dataset = dataset.map(_parse_function_train_fb_id, num_threads=4, output_buffer_size=1000)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(batch_size)
  return dataset

def get_eval_dataset_fb_id(file_paths, batch_size=1):
  dataset = tf.contrib.data.TFRecordDataset(file_paths)
  dataset = dataset.map(_parse_function_eval_fb_id, num_threads=4, output_buffer_size=2000)
  dataset = dataset.batch(batch_size)
  return dataset



