import os

from audioread import NoBackendError

import librosa
import numpy as np
import tensorflow as tf
import csv
import ipdb

def bytes_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a byte array.
  '''

  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
  '''
  Creates a TensorFlow Record Feature with value as a 64 bit integer.
  '''

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 44.1 Hz SR, with sailendcy label
def prepare_tfrecord_type6(example_paths, destination_path):

  with open('./Data/UrbanSound8K/metadata/UrbanSound8K.csv') as f:
    fb_list = list(csv.reader(f))
    fb_list = [[s[0].split('.')[0], s[4]] for s in fb_list]

  nfft = 2048
  whop = 512

  # Open a TFRecords file for writing.
  writer = tf.python_io.TFRecordWriter(destination_path)
  for idx in range(len(example_paths)):
    # Load an audio file for preprocessing.
    print('Extracting %s' % example_paths[idx])
    try:
      samples, sr = librosa.core.load(example_paths[idx], sr=44100)
      samples = (samples - samples.mean()) / (10*samples.std() + np.finfo(np.float).eps)
    except NoBackendError:
      print('Warning: Could not load {}.'.format(example_paths[idx]))
      continue

    spectrum_w512  = np.abs(librosa.core.stft(y=samples, n_fft=nfft, hop_length=whop, win_length=512))
    spectrum_w1024 = np.abs(librosa.core.stft(y=samples, n_fft=nfft, hop_length=whop, win_length=1024))
    spectrum_w1536 = np.abs(librosa.core.stft(y=samples, n_fft=nfft, hop_length=whop, win_length=1536))
    
    mel_basis             = librosa.filters.mel(sr, nfft)
    # T x F, logarithm of magnitude STFT
    logmelspectrum_w512   = np.transpose(np.log10(np.dot(mel_basis, spectrum_w512)+1).astype(np.float32))
    logmelspectrum_w1024  = np.transpose(np.log10(np.dot(mel_basis, spectrum_w1024)+1).astype(np.float32))
    logmelspectrum_w1536  = np.transpose(np.log10(np.dot(mel_basis, spectrum_w1536)+1).astype(np.float32))

    # T x F x C, (T, 128, 3)
    spectrum = np.stack((logmelspectrum_w512, logmelspectrum_w1024, logmelspectrum_w1536), -1)
 
    # sequence-level normalization, bin x frarmes, bin-wise normalization
    bin_mean = np.mean(spectrum,0)
    bin_std  = np.std(spectrum,0)
    spectrum = (spectrum - bin_mean) / bin_std

    label = int(os.path.split(example_paths[idx])[-1].split('-')[1])

    # for-back label
    wav_id   = os.path.split(example_paths[idx])[-1].split('.')[0].split('_')[0]
    label_fb = int([x[1] for x in fb_list if x[0] == wav_id][0])
    if label_fb == 2:
      label_fb = 0

    # Write the final spectrum and label to disk.
    example = tf.train.Example(features=tf.train.Features(feature={
      'spectrum': bytes_feature(spectrum.flatten().tostring()),
      'label': int64_feature(label),
      'label_fb': int64_feature(label_fb)
    }))
    writer.write(example.SerializeToString())
  writer.close()


## feature extraction, save into tfrecords
rent_dest = 'Data'

# original dataset
SOUND_FILE_DIRS = [
  'Data/UrbanSound8K/audio/fold1',
  'Data/UrbanSound8K/audio/fold2',
  'Data/UrbanSound8K/audio/fold3',
  'Data/UrbanSound8K/audio/fold4',
  'Data/UrbanSound8K/audio/fold5',
  'Data/UrbanSound8K/audio/fold6',
  'Data/UrbanSound8K/audio/fold7',
  'Data/UrbanSound8K/audio/fold8',
  'Data/UrbanSound8K/audio/fold9',
  'Data/UrbanSound8K/audio/fold10'
]

for fold in SOUND_FILE_DIRS:
  wav_paths = [os.path.join(fold, f) for f in os.listdir(fold) if f.endswith('.wav')]
  dest_path = os.path.join(parent_dest, 'urban_sound_' + fold.split('/')[3] +'.tfrecords6')
  prepare_tfrecord_type6(wav_paths, dest_path)

