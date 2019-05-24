# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
from six.moves import xrange 
# pylint: disable=redefined-builtin
import tensorflow as tf
# http://www.tensorfly.cn/        tf中文社区
# https://www.tensorflow.org/      英文社区

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops 
# 操作节点的IO
from tensorflow.python.platform import gfile
'''
tf.gfile文件读写函数
https://www.jianshu.com/p/d8f5357b95b3
https://blog.csdn.net/pursuit_zhangyu/article/details/80557958
'''
from tensorflow.python.util import compat
'''
Module: tf.compat
模块:tf.compat
Functions for Python 2 vs. 3 compatibility.
Python 2与Python 3兼容的函数。
see more:
https://tensorflow.google.cn/api_docs/python/tf/compat
'''

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


def prepare_words_list(wanted_words):   
# line 331附近使用了这个函数
  """Prepends common tokens to the custom word list.
	一共需要训练多少个命令，在你选的基础上加2，比如，你只选了up，那就是需要训练3个命令
	官网例子选了10个命令： 'yes,no,up,down,left,right,on,off,stop,go'，那就是需要训练12个命令
	返回命令的list

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
    List with the standard silence and unknown tokens added.
  """
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words
  # 返回要训练的命令标签list，加上了静音和未知两个标签


def which_set(filename, validation_percentage, testing_percentage):   
 # line 298附近使用了这个函数
  """Determines which data partition the file should belong to.
	判断这个文件属于哪个集合，是训练集，验证集，还是测试集
	返回一个字符串
	至于为什么要保证每次训练的时候，一个文件属于一个固定的集合，自己百度吧
	
  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result
  # 根据文件名（和各集合比例）判断该文件属于哪个集合，是训练集，验证集，还是测试集
  # 返回一个字符串，String, one of 'training', 'validation', or 'testing'.
  # 至于为什么要保证每次训练的时候，一个文件属于一个固定的集合，自己百度吧


def load_wav_file(filename):      
  """Loads an audio file and returns a float PCM-encoded array of samples.
	最终返回的是一个代表这段wav的Numpy array
  Args:
    filename: Path to the .wav file to load.

  Returns:
    Numpy array holding the sample data as floats between -1.0 and 1.0.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)  
	# 通过节点的操作来获取文件
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()
		'''
		wav_decoder是decode_wav函数得来的，decode_wav的作用是最终返回一串-1到1之间的数字，是一个tensor，
		decode_wav 的输入是一个16bit的wavfile：
		Decode a 16-bit PCM WAV file to a float tensor.
		The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.
		'''
	# 这个函数只是被定义了，在input_data.py和train.py和models.py中都没有被使用,在input_data_test.py中line150附近被使用
	# Returns:
	# Numpy array holding the sample data as floats between -1.0 and 1.0.

def save_wav_file(filename, wav_data, sample_rate):   
  """Saves audio sample data to a .wav audio file.

  Args:
    filename: Path to save the file to.
    wav_data: 2D array of float PCM-encoded audio data.
    sample_rate: Samples per second to encode in the file.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.placeholder(tf.string, [])
    sample_rate_placeholder = tf.placeholder(tf.int32, [])
    wav_data_placeholder = tf.placeholder(tf.float32, [None, 1])
    wav_encoder = contrib_audio.encode_wav(wav_data_placeholder,
                                           sample_rate_placeholder)
   '''
   encode_wav:
	   Encode audio data using the WAV file format.

	  This operation will generate a string suitable to be saved out to create a .wav

	  audio file. It will be encoded in the 16-bit PCM format. It takes in float

	  values in the range -1.0f to 1.0f, and any outside that value will be clamped to

	  that range.
   '''
    wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
	'''
	write_file:
	Writes contents to the file at input filename. Creates file and recursively
	creates directory if not existing.
	'''
    sess.run(
        wav_saver,
        feed_dict={
            wav_filename_placeholder: filename,
            sample_rate_placeholder: sample_rate,
            wav_data_placeholder: np.reshape(wav_data, (-1, 1))
        })
	# 这个函数跟上面函数的作用正好相反，也只是被定义了，暂时没有使用

def get_features_range(model_settings):  
  """Returns the expected min/max for generated features.
	返回特征的最大值和最小值，应该是为了归一化使用
  Args:
    model_settings: Information about the current model being trained.

  Returns:
    Min/max float pair holding the range of features.

  Raises:
    Exception: If preprocessing mode isn't recognized.
  """
  # TODO(petewarden): These values have been derived from the observed ranges
  # of spectrogram and MFCC inputs. If the preprocessing pipeline changes,
  # they may need to be updated.
  if model_settings['preprocess'] == 'average':
    features_min = 0.0
    features_max = 127.5
  elif model_settings['preprocess'] == 'mfcc':
    features_min = -247.0
    features_max = 30.0
  else:
    raise Exception('Unknown preprocess mode "%s" (should be "mfcc" or'
                    ' "average")' % (model_settings['preprocess']))
  return features_min, features_max
	# 这个函数在本文件中只是定义，没有使用，但在train.py的line135附近被使用了
	# 返回特征的最大值和最小值，应该是为了归一化使用

class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data.
	加载音频、分集合和准备音频训练数据
  """

  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
               wanted_words, validation_percentage, testing_percentage,
               model_settings, summaries_dir):
    if data_dir:  # 如果这个文件夹不为空
      self.data_dir = data_dir
      self.maybe_download_and_extract_dataset(data_url, data_dir)
      self.prepare_data_index(silence_percentage, unknown_percentage,
                              wanted_words, validation_percentage,
                              testing_percentage)
      self.prepare_background_data()
	# 无论data_dir是否为空，下面这个函数肯定是要执行的
    self.prepare_processing_graph(model_settings, summaries_dir)

  def maybe_download_and_extract_dataset(self, data_url, dest_directory):  
  # line 232附近使用了这个函数
    """Download and extract data set tar file.
	下载，并解压缩文件
    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.
	如果data_url这个字符串为空，则什么也不做，默认data_directory已经包含了正确的文件

    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url: # 如果data_url这个字符串为空，则什么也不做，这个函数就当不存在
      return
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        tf.logging.error('Failed to download URL: %s to folder: %s', data_url,
                         filepath)
        tf.logging.error('Please make sure you have enough free space and'
                         ' an internet connection')
        raise
      print()
      statinfo = os.stat(filepath)
      tf.logging.info('Successfully downloaded %s (%d bytes)', filename,
                      statinfo.st_size)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
	# 下载，并解压缩文件

  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage):                          
						 # line 233附近使用了这个函数
    """Prepares a list of the samples organized by set and label.
	应该返回是这三个东西
	self.data_index，
	self.words_list，
	self.word_to_index，

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.

    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
	  '''
	  wanted_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
	  wanted_words_index:
	  {'down': 5,
		 'go': 11,
		 'left': 6,
		 'no': 3,
		 'off': 9,
		 'on': 8,
		 'right': 7,
		 'stop': 10,
		 'up': 4,
		 'yes': 2}
	  '''
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.data_dir, '*', '*.wav')
	
	#下面从35个文件夹中选出我们要的训练集，验证集和测试集，都保存在self.data_index这个字典中
    for wav_path in gfile.Glob(search_path):
      _, word = os.path.split(os.path.dirname(wav_path))
      word = word.lower()
	  # word 是那个文件夹中的up，down，go等36个命令的单词
	  
      # Treat the '_background_noise_' folder as a special case, since we expect
      # it to contain long audio samples we mix in to improve training.
      if word == BACKGROUND_NOISE_DIR_NAME:
	  # BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
        continue
      all_words[word] = True
	  # all_words是一个dict，每一个key对应的value都是true
	  
      set_index = which_set(wav_path, validation_percentage, testing_percentage) 
	  # set_index，文件所属集合的名字，是一个字符串，'validation' or 'testing' or 'training'
	  # 根据原则，每个文件在特定的比例下永远只属于一个集合，是由文件名字决定的
	  
      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      if word in wanted_words_index:
	  # 假如现在 word ='up'，在wanted_words_index这个字典的key中：
        self.data_index[set_index].append({'label': word, 'file': wav_path})
		# self.data_index是一个dict， = {'validation': [], 'testing': [], 'training': []}
		# self.data_index的每一个key对应的value首先是一个list，这个list中的各个元素也是一个dict，小dict有俩key
		# 一个是文件对应的命令label，这里label对应的word就是文件夹的名字，一个是文件的绝对地址wav_path
		# 如：self.data_index['validation'][0]['file']就是验证集中第一个文件的绝对地址
		# self.data_index['validation'][0]['label']就是验证集中第一个文件对应的label
      else:
	  # 假如现在 word ='wow'，不在wanted_words_index这个字典的key中：
        unknown_index[set_index].append({'label': word, 'file': wav_path})
		# unknown_index也是一个dict，结构和self.data_index几乎一摸一样
		# unknown_index也有三个key，'validation' ，'testing' ，'training'，因为每个文件对应唯一的集合，也就是对应唯一的key
		
    if not all_words:
	# 如果all_words为空：
      raise Exception('No .wavs found at ' + search_path)
	  
    for index, wanted_word in enumerate(wanted_words):
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
						
	# 下面给训练集，验证集和测试集都加入相同比例的静音音频，更新self.data_index这个字典
    # We need an arbitrary任意的 file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
	  # set_size 集合的大小
	  # 每个集合中都要有一定比例的静音，很简单，把数值设为0就可以了
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
	  # math.ceil(x)返回大于等于参数x的最小整数,即对浮点数向上取整
	  # 假设对于验证集，silence_size = 4，也就是加入4个静音音频
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
		# 这里用了append的方法，下面用了extend的方法
		
		
	  # 下面给训练集，验证集和测试集都加入相同比例的未知label的音频，再次更新self.data_index这个字典
      # Pick some unknowns to add to each partition of the data set.
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
	  #假设对于训练集，unknown_size=10，也就加入10个未知label的音频
      self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
	  
	  
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
	  
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(wanted_words)
	# self.words_list =['_silence_','_unknown_','yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go]
    
	
	self.word_to_index = {}
    for word in all_words:
	# all_words的key包含了35个单词，对应的value都是true
      if word in wanted_words_index:
	  # wanted_words_index 见line334，是一个有10个key的dict
        self.word_to_index[word] = wanted_words_index[word]
		# self.word_to_index也是一个dict，比如，yes在wanted_words_index中的value是2，
		# 那么yes在self.word_to_index这个dict中对应的value也是2
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
		# UNKNOWN_WORD_INDEX 预先被定义为1
		# 比如 wow在all_words的key中，但是不在wanted_words_index的key中
		# 那么，wow在self.word_to_index这个dict中对应的value就是1
		# 也就是说，self.word_to_index这个dict的key也是35个单词，但是value却不一样
		#SILENCE_LABEL = '_silence_'
		#SILENCE_INDEX = 0
		#在下面这句话之前，self.word_to_index有35个item，加入SILENCE_LABEL之后是36个item
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX
	#整体来说这个函数应该返回是这三个东西：
	#self.data_index，self.data_index = {'validation': [], 'testing': [], 'training': []}
		#包含了从35个文件夹中选出来的，我们需要的训练集，验证集和测试集
	#self.words_list，
		#self.words_list =['_silence_','_unknown_','yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go]
	#self.word_to_index，
		#self.word_to_index这个dict中有36个key（35个已知的命令label加一个静音lable）
		#不在wanted_words里面的key对应的value都是1，在里面的key的value见line334
		#静音对应的value是0
	
  def prepare_background_data(self):   
  # line 236附近使用了这个函数
    """Searches a folder for background noise audio, and loads it into memory.
	返回的是self.background_data ，它是一个list，因为有好几种背景噪音，故list中又有好几个小list，
	每个小list都是一串数字，代表一种噪音
    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.
	  也即是：self.background_data 

    Raises:
      Exception: If files aren't found in the folder.
    """
    self.background_data = []
    background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
      return self.background_data
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
      search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                                 '*.wav')
      for wav_path in gfile.Glob(search_path):
        wav_data = sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
        self.background_data.append(wav_data)
      if not self.background_data:
        raise Exception('No background wav files were found in ' + search_path)
	#返回值是self.background_data ，一个包含各种噪音的list
	
  def prepare_processing_graph(self, model_settings, summaries_dir):
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file，加载一段音频, decodes it解码, scales the volume调整音量的大小,
    shifts it in time, adds in background noise加入背景噪音, calculates a spectrogram计算出一个频谱图, and
    then builds an MFCC fingerprint from that，然后计算出MFCC值.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - output_: Output 2D fingerprint of processed audio.

    Args:
      model_settings: Information about the current model being trained.
      summaries_dir: Path to save training summary information to.
	  
	返回值应该有下面三个，但最主要的还是self.output_，它代表了最终输入神经网络的特征
	self.output_ 
	self.merged_summaries_
	self.summary_writer_ 

    Raises:
      ValueError: If the preprocessing mode isn't recognized.
    """
    with tf.get_default_graph().name_scope('data'):
      desired_samples = model_settings['desired_samples']
      self.wav_filename_placeholder_ = tf.placeholder(
          tf.string, [], name='wav_filename')
      wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
      wav_decoder = contrib_audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
		  '''
			decode_wav函数的说明是这个样子的：
			def decode_wav(contents, desired_channels=-1, desired_samples=-1, name=None):
			
			Decode a 16-bit PCM WAV file to a float tensor.
			The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.
			Args:
			contents: A `Tensor` of type `string`.
					The WAV-encoded audio, usually from a file.
			desired_channels: An optional `int`. Defaults to `-1`.
					Number of sample channels wanted.
			desired_samples: An optional `int`. Defaults to `-1`.
					Length of audio requested.
			name: A name for the operation (optional).

			Returns:
			A tuple of `Tensor` objects (audio, sample_rate).

			audio: A `Tensor` of type `float32`.
			sample_rate: A `Tensor` of type `int32`
			
			这说明wav_decoder是一组-1到1之间的float32型的数字，是一个tensor，表示一段wav
		  '''
		  
      # Allow the audio sample's volume to be adjusted.
      self.foreground_volume_placeholder_ = tf.placeholder(
          tf.float32, [], name='foreground_volume')
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                      self.foreground_volume_placeholder_)
		'''
		上面这句话是不是说明对一段wav进行音量的调节就是将表示这段wav的数字矩阵乘以音量矩阵？
		'''
      # Shift the sample's start position, and pad any gaps with zeros.
      self.time_shift_padding_placeholder_ = tf.placeholder(
          tf.int32, [2, 2], name='time_shift_padding')
      self.time_shift_offset_placeholder_ = tf.placeholder(
          tf.int32, [2], name='time_shift_offset')
      padded_foreground = tf.pad(
          scaled_foreground,
          self.time_shift_padding_placeholder_,
          mode='CONSTANT')
      sliced_foreground = tf.slice(padded_foreground,
                                   self.time_shift_offset_placeholder_,
                                   [desired_samples, -1])
      # Mix in background noise.
      self.background_data_placeholder_ = tf.placeholder(
          tf.float32, [desired_samples, 1], name='background_data')
      self.background_volume_placeholder_ = tf.placeholder(
          tf.float32, [], name='background_volume')
      background_mul = tf.multiply(self.background_data_placeholder_,
                                   self.background_volume_placeholder_)
      background_add = tf.add(background_mul, sliced_foreground)
	  '''
		上面这两句话的意思是不是先对背景噪音进行音量调节，再将调节过音量大小的背景噪音加入到切片过的wav中？
		这样就实现了对一段wav进行混入背景噪音的处理么？
		'''
      background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
	  '''
	  tf.clip_by_value(A, min,max): 输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
	  
	  clamp: vt. 夹紧，固定住 n. 夹钳，螺丝钳
	  '''
      # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
      spectrogram = contrib_audio.audio_spectrogram(
          background_clamp,
          window_size=model_settings['window_size_samples'],
          stride=model_settings['window_stride_samples'],
          magnitude_squared=True)
		  '''
		  上面这句话是生成一段wav的频谱图
		  '''
      tf.summary.image(
          'spectrogram', tf.expand_dims(spectrogram, -1), max_outputs=1)
		  '''
		  上面这句话与tensorflow的可视化有关，可能要显示在tensorboard中吧。
		  
		  TensorFlow中，想要维度增加一维，可以使用tf.expand_dims(input, dim, name=None)函数。当然，我们常用tf.reshape(input, shape=[])也可以达到相同效果，但是有些时候在构建图的过程中，placeholder没有被feed具体的值，这时就会包下面的错误：TypeError: Expected binary or unicode string, got 1 
			在这种情况下，我们就可以考虑使用expand_dims来将维度加1。比如我自己代码中遇到的情况，在对图像维度降到二维做特定操作后，要还原成四维[batch, height, width, channels]，前后各增加一维。
			
			# 't' is a tensor of shape [2]
				shape(expand_dims(t, 0)) ==> [1, 2]
				shape(expand_dims(t, 1)) ==> [2, 1]
				shape(expand_dims(t, -1)) ==> [2, 1]

				# 't2' is a tensor of shape [2, 3, 5]
				shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
				shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
				shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
		  '''
      # The number of buckets in each FFT row in the spectrogram will depend on
      # how many input samples there are in each window. This can be quite
      # large, with a 160 sample window producing 127 buckets for example. We
      # don't need this level of detail for classification, so we often want to
      # shrink them down to produce a smaller result. That's what this section
      # implements. One method is to use average pooling to merge adjacent
      # buckets, but a more sophisticated approach is to apply the MFCC
      # algorithm to shrink the representation.
      if model_settings['preprocess'] == 'average':
        self.output_ = tf.nn.pool(
            tf.expand_dims(spectrogram, -1),
            window_shape=[1, model_settings['average_window_width']],
            strides=[1, model_settings['average_window_width']],
            pooling_type='AVG',    # 平均池化
            padding='SAME')
			'''
		  上面这句话的意思应该是，如果preprocess选的是average的话，那么输入神经网络的特征是池化后的频谱图，
		  因为下面有个shrunk_spectrogram，也就spectrogram变小了
		  '''
        tf.summary.image('shrunk_spectrogram', self.output_, max_outputs=1)
      elif model_settings['preprocess'] == 'mfcc':
        self.output_ = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=model_settings['fingerprint_width'])
			'''
		  上面这句话计算mfcc，如果preprocess选的是mfcc的话，那么输入神经网络的特征就是计算出来的mfcc，
		  fingerprint_width是最开始的参数，选的是40，那就是40维的mfcc
		  其实spectrogram和mfcc值都是一串数字
		  '''
        tf.summary.image(
            'mfcc', tf.expand_dims(self.output_, -1), max_outputs=1)
      else:
        raise ValueError('Unknown preprocess mode "%s" (should be "mfcc" or'
                         ' "average")' % (model_settings['preprocess']))

      # Merge all the summaries and write them out to /tmp/retrain_logs (by
      # default)
      self.merged_summaries_ = tf.summary.merge_all(scope='data')
      if summaries_dir:
        self.summary_writer_ = tf.summary.FileWriter(summaries_dir + '/data',
                                                     tf.get_default_graph())
	#准备计算图，或者过程图
	#整体来看，这个函数的返回值应该有下面三个，但最主要的还是self.output_，它代表了最终输入神经网络的特征
	#self.output_ 
	#self.merged_summaries_
	#self.summary_writer_ 
	
  def set_size(self, mode):   
  # line347,348,356都有使用这个函数
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])
	# 返回值是len(self.data_index[mode])，集合的大小
	# self.data_index = {'validation': [], 'testing': [], 'training': []}

  def get_data(self, how_many, offset, model_settings, background_frequency,
               background_volume_range, time_shift, mode, sess):
    """Gather samples from the data set, applying transformations as needed.
	这个函数主要用来获取每个batch的样本，并进行加噪变换
	
    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition. -1意味着整个集合都被选用
		how_many其实是你自己定义的每个batch用的样本数量
      offset: Where to start when fetching deterministically.
	  offset用来标识每个batch从哪里开始
      model_settings: Information about the current model being trained.
      background_frequency: How many clips will have background noise, 0.0 to 1.0. 假如要在一个batch的训练集合中加入噪音，那么是所有的样本音频都被加入噪音么，还是只有一部分？
      background_volume_range: How loud the background noise will be.
	  加入的背景噪音的音量有多大，当然噪音越大，识别越困难啊
      time_shift: How much to randomly shift the clips by in time.
	  这个time shift，还没搞清楚它是做什么用的
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.
		mode是一个字符串，只能是上面三者之一
      sess: TensorFlow session that was active when processor was created.

    Returns:
      List of sample data for the transformed samples, and list of label indexes

    Raises:
      ValueError: If background samples are too short.
    """
    # Pick one of the partitions to choose samples from.
    candidates = self.data_index[mode]
	# self.data_index = {'validation': [], 'testing': [], 'training': []}
    if how_many == -1:
	# -1的意思是全选
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
	  # sample_count是每个batch的音频数量，假如你定义的batchsize是64，那么sample_count大部分情况下是64，
	  # 假如len(candidates)=3000，那么最后一个batch是56
    # Data and labels will be populated and returned.
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
	# data 是一个64行，3920列的全0矩阵，shape是(64,3920)
    labels = np.zeros(sample_count)
	# labels 应该是一个含有64个0的数组，它的shape是(64,)，而不是(64,1),也就是说它不是一个矩阵，吴恩达老师不建议我们使用数组，
	# 因为后期计算的话还要reshape，不如直接生成64行1列的矩阵比较好，吴老师还建议batchsize是2的n次方比较好
    desired_samples = model_settings['desired_samples']
	# desired_samples =16000
    use_background = self.background_data and (mode == 'training')
	# 当self.background_data（这是一个list）不为空且后面等式为true的时候，use_background为true，
	# 当后面等式为false，use_background为false
	# 当self.background_data为空的时候，不管后面是否成立，use_background是一个空list
    pick_deterministically = (mode != 'training')
	# 当mode不为'training'时候，pick_deterministically为true
	
    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
	  # python3中直接使用range就可以
      # Pick which audio sample to use.
      if how_many == -1 or pick_deterministically:
        sample_index = i
      else:
        sample_index = np.random.randint(len(candidates))
      sample = candidates[sample_index]
	  # sample 应该是个dict结构，key有file，label之类
	  # 上面的意思是如果使用集合中的全部数据，或者pick_deterministically为true的时候，sample_index是顺序的，一个挨着一个的
	  # 否则， sample_index就是在candidates中被随机抽取的，train.py中使用了三次get_data函数
		# 231行，训练集中使用，offset为0
		# 270行，验证集中使用，offset为0
		# 305行，测试集中使用，offset为i，应该是随机选取一些sample进行测试的，测试集不是全部被使用
		
      # If we're time shifting, set up the offset for this sample.
      if time_shift > 0:
	  # time_shift = 1600
        time_shift_amount = np.random.randint(-time_shift, time_shift)
		# 假如 time_shift_amount = 1309和-1309
      else:
        time_shift_amount = 0
      if time_shift_amount > 0:
        time_shift_padding = [[time_shift_amount, 0], [0, 0]]
        time_shift_offset = [0, 0]
		# time_shift_amount = 1309时，time_shift_padding = [[1309,0], [0, 0]]，time_shift_offset = [0, 0]
      else:
        time_shift_padding = [[0, -time_shift_amount], [0, 0]]
        time_shift_offset = [-time_shift_amount, 0]
		#  time_shift_amount = -1309时，time_shift_padding = [[0, 1309], [0, 0]]，time_shift_offset =[1309, 0]
		
      input_dict = {
          self.wav_filename_placeholder_: sample['file'],
		  # sample['file']应该是wav文件的名字
          self.time_shift_padding_placeholder_: time_shift_padding,
          self.time_shift_offset_placeholder_: time_shift_offset,
      }
	  
      # Choose a section of background noise to mix in.
      if use_background or sample['label'] == SILENCE_LABEL:
	  # 若果使用加噪，后者sample的lable是静音，对应的else在line 737
	  # or的意思是静音是一定加噪的是么？
        background_index = np.random.randint(len(self.background_data))
			# 随机选择一种噪音的序号
        background_samples = self.background_data[background_index]
		# 选择这种噪音，background_samples是一个list，里面是一串-1到1之间的数字
        if len(background_samples) <= model_settings['desired_samples']:
		# model_settings['desired_samples']=16000，也就是说如果噪音的采样率小于或等于每秒16000次的话，raise error
		# 假设len(background_samples)=16500
          raise ValueError(
              'Background sample is too short! Need more than %d'
              ' samples but only %d were found' %
              (model_settings['desired_samples'], len(background_samples)))
			  
        background_offset = np.random.randint(
            0, len(background_samples) - model_settings['desired_samples'])
			# 假如background_offset =102
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
			# line 650 ：desired_samples=16000
			# background_clipped也是一串数字组成的list，从offset102开始，到102+16000，长度也是16000
			# 但是万一offset选的有点大，一直选到末尾了还不够16000，那下面reshape就会有问题了
			# 我们选的16500是大于102+16000的，所以不会有问题
		
        background_reshaped = background_clipped.reshape([desired_samples, 1])
		# 这句应该是把tensor数组reshape成矩阵，reshape要求维度的乘积是一样的，否则不可以reshape
		# background_reshaped就是我们从噪音中选出来的和sample长度一致的一小段
        if sample['label'] == SILENCE_LABEL:
          background_volume = np.random.uniform(0, 1)
		  # 假设这里background_volume =0.3473247211726097
        elif np.random.uniform(0, 1) < background_frequency:
		# 如果不是SILENCE_LABEL，假如是up_label或者是其它我们需要训练的命令,这里重新产生了一个随机数字
		# background_frequency其实就是train.py中的FLAGS.background_frequency=0.8
          background_volume = np.random.uniform(0, background_volume_range)
		  # background_volume_range其实就是train.py中的FLAGS.background_volume=0.1
		  # 假设这里background_volume = 0.040985952853297924
        else:
          background_volume = 0
		  # 也就是说，如果要加入噪音，每一个样本加入噪音的音量都不同，因为随机数每次都不一样
		  # 也有可能某一小部分样本加入的噪音音量为0
		  # 只不过整体来说，静音样本的噪音音量要比非静音样本的音量大一点点
      else:
	  # 对应的if在 line 699，就是说如果不加噪音的话，直接把background_reshaped和background_volume都设置为0
        background_reshaped = np.zeros([desired_samples, 1])
		# background_reshaped，这里直接就生成了一个array，而不是数组
        background_volume = 0
		
      input_dict[self.background_data_placeholder_] = background_reshaped
	  # 给字典加键值对
      input_dict[self.background_volume_placeholder_] = background_volume
	  
      # If we want silence, mute out the main sample but leave the background.
      if sample['label'] == SILENCE_LABEL:
        input_dict[self.foreground_volume_placeholder_] = 0
		# 给字典加键值对，设置前景音
      else:
        input_dict[self.foreground_volume_placeholder_] = 1
		
      # Run the graph to produce the output audio.
      summary, data_tensor = sess.run(
          [self.merged_summaries_, self.output_], feed_dict=input_dict)
		  # sess.run 有几个过程，就有几个结果
		  # 最终的input_dict最终有6个key，大概就是line 883那个样子
		  # self.merged_summaries_在line 445和585都出现过
		  # self.output_ 在line 556和568，有定义，应该是mfcc的值或者是频谱图的值
		  
		  
      self.summary_writer_.add_summary(summary)
      data[i - offset, :] = data_tensor.flatten()
	  # data 在line 645 有定义，是一个64行，3920列的矩阵，是一个batch的特征矩阵
	  # data的每一行是一个样本的特征，data_tensor得到的是一个样本的特征，flatten之后放入data矩阵
      label_index = self.word_to_index[sample['label']]
      labels[i - offset] = label_index
	  # 同理，把这一行样本对应的label标签也加入labels，line647有定义
    return data, labels
	# 整体来说，get_data函数是返回一个batch的特征矩阵和标签，矩阵中的每一行样本都已经进行了加噪处理
	# 但是我怎么感觉还是没有进行加噪啊，只是定义了各种噪音的音量大小啊
	# time_shift最后被搞在了input_dict里面，具体是干啥的，还是不知道啊

  def get_features_for_wav(self, wav_filename, model_settings, sess):
    """Applies the feature transformation process to the input_wav.

    Runs the feature generation process (generally producing a spectrogram from
    the input samples) on the WAV file. This can be useful for testing and
    verifying implementations being run on other platforms.

    Args:
      wav_filename: The path to the input audio file.
      model_settings: Information about the current model being trained.
      sess: TensorFlow session that was active when processor was created.

    Returns:
      Numpy data array containing the generated features.
    """
    desired_samples = model_settings['desired_samples']
	# desired_samples = 16000
    input_dict = {
        self.wav_filename_placeholder_: wav_filename,
        self.time_shift_padding_placeholder_: [[0, 0], [0, 0]],
        self.time_shift_offset_placeholder_: [0, 0],
        self.background_data_placeholder_: np.zeros([desired_samples, 1]),
        self.background_volume_placeholder_: 0,
        self.foreground_volume_placeholder_: 1,
    }
    # Run the graph to produce the output audio.
    data_tensor = sess.run([self.output_], feed_dict=input_dict)
    return data_tensor
	# 这个函数在train.py，input_data.py和models.py中都没有被使用过
	# 返回值是：Numpy data array containing the generated features.

  def get_unprocessed_data(self, how_many, model_settings, mode):
    """Retrieve sample data for the given partition, with no transformations.
	检索给定集合的样本，不进行加噪处理
    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      model_settings: Information about the current model being trained.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.

    Returns:
      List of sample data for the samples, and list of labels in one-hot form.
    """
    candidates = self.data_index[mode]
	# self.data_index = {'validation': [], 'testing': [], 'training': []}
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = how_many
    desired_samples = model_settings['desired_samples']
	# desired_samples =16000
    words_list = self.words_list
	# words_list =['_silence_','_unknown_','yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go]
    data = np.zeros((sample_count, desired_samples))
	# data 是sample_count行，16000列的全0矩阵
    labels = []
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = contrib_audio.decode_wav(
          wav_loader, desired_channels=1, desired_samples=desired_samples)
      foreground_volume_placeholder = tf.placeholder(tf.float32, [])
	  # 前景音音量foreground_volume
      scaled_foreground = tf.multiply(wav_decoder.audio,
                                      foreground_volume_placeholder)
	# 调整过音量的前景声音:scaled_foreground，是由一串数字乘以一个数字得到的
	# wav_decoder 是由decode_wav这个函数得到的，可能是一个dict结构，
	# 有一个key为audio，value为一串-1到1之间的数字，代表这段声音
	# 其它的key可能是声道channels啊什么的，因为看不懂decode_wav的返回值，纯猜测的
	
      for i in range(sample_count):
        if how_many == -1:
          sample_index = i
        else:
          sample_index = np.random.randint(len(candidates))
        sample = candidates[sample_index]
		# 上面的意思是在集合中随机挑选样本
		# 下面开始完善这个函数中的input_dict
        input_dict = {wav_filename_placeholder: sample['file']}
        if sample['label'] == SILENCE_LABEL:
		# sample 应该是个dict结构，key有file，label之类
          input_dict[foreground_volume_placeholder] = 0
        else:
          input_dict[foreground_volume_placeholder] = 1
		  
        data[i, :] = sess.run(scaled_foreground, feed_dict=input_dict).flatten()
		# 上面这句话的意思应该是对data的第i行进行前景音量的调节，最后flatten
        label_index = self.word_to_index[sample['label']]
        labels.append(words_list[label_index])
		# words_list =['_silence_','_unknown_','yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go]
    return data, labels
	#返回值是data，labels
	#data是sample_count行，16000列的数字矩阵
	#每一行是调整过前景音音量的音频，是一串-1到1 之间的数字
	#labels是一个list，每一行音频对应的label
	#这个函数只是被定义了，在input_data.py和models.py还有train.py中没有被使用
