# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2
import random
import math
import tensorflow as tf
import visualize

############ 定数
NUM_CLASSES = 6 # 分類するクラス数
IMG_SIZE = 28 # 画像の1辺の長さ
COLOR_CHANNELS = 3 # RGB
IMG_PIXELS = IMG_SIZE * IMG_SIZE * COLOR_CHANNELS # 画像のサイズ*RGB
STEPS = 100 # 学習ステップ数
BATCH_SIZE = 50 # バッチサイズ
TRAIN_IMG_BASE_DIR = '../images/train_28/' # 訓練データ格納ディレクトリ

############ Functions
# データ取得
def get_train_data(data_dir):
    # 画像のあるディレクトリ
    img_dirs = ['0_nemu', '1_risa', '2_mirin', '3_ei', '4_pin', '5_moga']

    # 学習画像データ
    train_image = []
    # 学習データのラベル
    train_label = []

    for i, d in enumerate(img_dirs):
        # ./images/以下の各ディレクトリ内のファイル名取得
        files = os.listdir(data_dir + '/' + d)
        for f in files:
            # 画像読み込み
            path, ext = os.path.splitext(f)
            if ext != ".jpg":
                continue
            img = cv2.imread(TRAIN_IMG_BASE_DIR + d + '/' + f)
            # img = cv2.imread('./images/train/' + d + '/' + f)
            # 1辺がIMG_SIZEの正方形にリサイズ
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # 1列にして
            img = img.flatten().astype(np.float32)/255.0
            train_image.append(img)

            # one_hot_vectorを作りラベルとして追加
            tmp = np.zeros(NUM_CLASSES)
            tmp[i] = 1
            train_label.append(tmp)

    # numpy配列に変換
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)

    return train_image, train_label


# 学習結果を可視化
def visualize_weight_data(step, w1, c1, p1, w2, c2, p2):
    visualize.init()

    # print('w1.shape=%s' % str(w1.shape))
    visualize.visualize_weights_1('step_%d_w1' % step, w1)

    # print('c1.shape=%s' % str(c1.shape))
    visualize.visualize_conv_1('step_%d_c1' % step, c1)

    # print('p1.shape=%s' % str(p1.shape))
    visualize.visualize_pool_1('step_%d_p1' % step, p1)

    # print('w2.shape=%s' % str(w2.shape))
    visualize.visualize_weights_2('step_%d_w2' % step, w2)

    # print('c2.shape=%s' % str(c2.shape))
    visualize.visualize_conv_2('step_%d_c2' % step, c2)

    # print('p2.shape=%s' % str(p2.shape))
    visualize.visualize_pool_2('step_%d_p2' % step, p2)


############ TensorFlow Graph
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

def inference(x, keep_prob):
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
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, NUM_CLASSES], 'W_fc2')
    b_fc2 = bias_variable([NUM_CLASSES], 'b_fc2')

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return W_conv1, h_conv1, h_pool1, W_conv2, h_conv2, h_pool2, W_fc1, h_fc1, y_conv


############ TensorFlow Session
def train(data_dir):
    # Placeholder
    x = tf.placeholder(tf.float32, shape=[None, IMG_PIXELS])
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)

    # Train and Evaluate the Model
    wconv1, hconv1, hpool1, wconv2, hconv2, hpool2, wfc1, hfc1, y_conv = inference(x, keep_prob)
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10,1.0)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ############ Session Start
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init_op)

        train_image, train_label = get_train_data(TRAIN_IMG_BASE_DIR)

        for i in range(STEPS):
            random_seq = list(range(len(train_image)))
            random.shuffle(random_seq)
            for j in range(math.floor(len(train_image)/BATCH_SIZE)):
                batch = BATCH_SIZE * j
                train_image_batch = []
                train_label_batch = []
                for k in range(BATCH_SIZE):
                    train_image_batch.append(train_image[random_seq[batch + k]])
                    train_label_batch.append(train_label[random_seq[batch + k]])

            _, w1, c1, p1, w2, c2, p2, w3, fc, yconv = sess.run([train_step, wconv1, hconv1, hpool1, wconv2, hconv2, hpool2, wfc1, hfc1, y_conv],
                feed_dict={x: train_image_batch, y_: train_label_batch, keep_prob: 0.5})

            # 学習データに対する正答率を表示
            if i % 1 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                  x:train_image, y_: train_label, keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))

            if i % (STEPS-1) == 0:
                # 学習結果を可視化
                visualize_weight_data(i, w1, c1, p1, w2, c2, p2)

        # 学習結果をsave
        saver.save(sess, './ckpt/train.ckpt')


if __name__ == "__main__":
    argvs = sys.argv
    if (len(argvs) != 2):
        print('Usage: # python %s <data_dir>' % argvs[0])
        quit()

    data_dir = argvs[1]
    train(data_dir)
