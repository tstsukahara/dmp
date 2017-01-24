
import os
import numpy as np
import cv2
import random
import math

import tensorflow as tf

# 定数
NUM_CLASSES = 6 # 分類するクラス数
IMG_SIZE = 28 # 画像の1辺の長さ
COLOR_CHANNELS = 3 # RGB
IMG_PIXELS = IMG_SIZE * IMG_SIZE * COLOR_CHANNELS # 画像のサイズ*RGB


sess = tf.InteractiveSession()

# 画像のあるディレクトリ
predict_img_dirs = ['0_nemu', '1_risa', '2_mirin', '3_ei', '4_pin', '5_moga']

# 学習画像データ
predict_image = []
# 学習データのラベル
predict_label = []

for i, d in enumerate(predict_img_dirs):
    # ./images/以下の各ディレクトリ内のファイル名取得
    files = os.listdir('./images/train/' + d)
    for f in files:
        # 画像読み込み
        path, ext = os.path.splitext(f)
        if ext != ".jpg":
            continue
        img = cv2.imread('./images/train/' + d + '/' + f)
        # 1辺がIMG_SIZEの正方形にリサイズ
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # 1列にして
        img = img.flatten().astype(np.float32)/255.0
        predict_image.append(img)

        # one_hot_vectorを作りラベルとして追加
        tmp = np.zeros(NUM_CLASSES)
        tmp[i] = 1
        predict_label.append(tmp)

# numpy配列に変換
predict_image = np.asarray(predict_image)
predict_label = np.asarray(predict_label)

# Model
x = tf.placeholder(tf.float32, shape=[None, IMG_PIXELS])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

# Weight Initialization
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

# Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# First Convolutional Layer
# サイズ5×5、3色, 32チャネルのフィルタ
W_conv1 = weight_variable([5, 5, COLOR_CHANNELS, 32], 'W_conv1')
b_conv1 = bias_variable([32], 'b_conv1')

# 1列で入力した画像データを3次元構造に戻す
x_image = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
b_conv2 = bias_variable([64], 'b_conv2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Density Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024], 'W_fc1')
b_fc1 = bias_variable([1024], 'b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, NUM_CLASSES], 'W_fc2')
b_fc2 = bias_variable([NUM_CLASSES], 'b_fc2')

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Predict
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
saver.restore(sess, "./ckpt/train.ckpt")

predict_accuracy = accuracy.eval(feed_dict={
  x:predict_image, y_: predict_label, keep_prob: 1.0})
print("predict accuracy %g" % predict_accuracy)
