
import os
import numpy as np
import cv2
import random
import math
from PIL import Image


import tensorflow as tf

# 定数
NUM_CLASSES = 6 # 分類するクラス数
IMG_SIZE = 28 # 画像の1辺の長さ
COLOR_CHANNELS = 3 # RGB
IMG_PIXELS = IMG_SIZE * IMG_SIZE * COLOR_CHANNELS # 画像のサイズ*RGB


sess = tf.InteractiveSession()

# 画像のあるディレクトリ
train_img_dirs = ['0_nemu', '1_risa', '2_mirin', '3_ei', '4_pin', '5_moga']

# 学習画像データ
train_image = []
# 学習データのラベル
train_label = []

for i, d in enumerate(train_img_dirs):
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
        train_image.append(img)

        # one_hot_vectorを作りラベルとして追加
        tmp = np.zeros(NUM_CLASSES)
        tmp[i] = 1
        train_label.append(tmp)

# numpy配列に変換
train_image = np.asarray(train_image)
train_label = np.asarray(train_label)

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

# For Weights Visualization
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

# Train and Evaluate the Model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init_op = tf.initialize_all_variables()
saver = tf.train.Saver()
sess.run(init_op)

STEPS = 10 # 学習ステップ数
BATCH_SIZE = 50 # バッチサイズ

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

        train_step.run(feed_dict={x: train_image_batch, y_: train_label_batch, keep_prob: 0.5})

    # 学習データに対する正答率を表示
    if i % 1 == 0:
        train_accuracy = accuracy.eval(feed_dict={
          x:train_image, y_: train_label, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

# 学習結果をsave
saver.save(sess, './ckpt/train.ckpt')

FILTER_COUNT = 32
GRID_SIZE_WIDTH = 4
GRID_SIZE_HEIGHT = 8

# フィルタを可視化
weights_data = W_conv1.eval()
# weights_data = h_pool1.eval()
print(weights_data.shape)

image_data = _color(weights_data)
image_data = _make_padding(image_data)

rows = None

# FILTER_COUNT 個のフィルターをグリッドに整形
for index in range(GRID_SIZE_HEIGHT):
    print('index=%s' % index)
    start = index * GRID_SIZE_WIDTH
    end = start + GRID_SIZE_WIDTH
    print('start=%s' % start)
    print('end=%s' % end)

    row = np.hstack(image_data[start:end])
    print(row.shape)
    if rows is None:
        rows = row
    else:
        rows = np.vstack((rows, row))
        print(rows.shape)


file_path = os.path.join('visualized_weights') + '.bmp'
with open(file_path, mode='wb') as fp:
    Image.fromarray(rows).save(fp, format='bmp')