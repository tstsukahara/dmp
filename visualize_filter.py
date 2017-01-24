# coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from PIL import Image

import numpy as np

# フィルターを可視化するスクリプト

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('graph_file', None, "処理するグラフファイルのパス")

FILTER_COUNT = 64

GRID_SIZE_WIDTH = 8
GRID_SIZE_HEIGHT = 8


def visualize(graph_file):
  basename = os.path.basename(graph_file)
  path = os.path.dirname(graph_file)

  with tf.gfile.FastGFile(graph_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
      for node in sess.graph_def.node:
        print(node.name)

      weights = sess.graph.get_tensor_by_name('conv1_1/weights:0')
      weights_data = weights.eval()

      image_data = _color(weights_data)
      image_data = _make_padding(image_data)

      rows = None

      # FILTER_COUNT 個のフィルターをグリッドに整形
      for index in range(GRID_SIZE_HEIGHT):
        start = index * GRID_SIZE_WIDTH
        end = start + GRID_SIZE_WIDTH

        row = np.hstack(image_data[start:end])
        if rows is None:
          rows = row
        else:
          rows = np.vstack((rows, row))

      print(rows.shape)

      file_path = os.path.join(path, basename) + '.bmp'
      with open(file_path, mode='wb') as fp:
        Image.fromarray(rows).save(fp, format='bmp')


def _color(weights_data):
  x_min = np.amin(weights_data)
  x_max = np.amax(weights_data)
  weights_0_to_1 = (weights_data - x_min) / (x_max - x_min)
  weights_0_to_255_uint8 = (weights_0_to_1 * 255).astype(np.uint8)
  image_data = np.transpose(weights_0_to_255_uint8, [3, 0, 1, 2])
  return image_data


def _make_padding(image_data):
  list = []
  for data in image_data:
    data = np.pad(data, pad_width=((1, 1), (1, 1), (0, 0)),
                  mode='constant', constant_values=0)
    list.append(data)
  return list


if __name__ == '__main__':
  visualize(FLAGS.graph_file)
