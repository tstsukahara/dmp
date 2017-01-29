# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
from PIL import Image

# For Weights Visualization
def visualize_weights_1(name, weights_data):
    # 値を0-255に変換
    x_min = np.amin(weights_data)
    x_max = np.amax(weights_data)
    weights_0_to_1 = (weights_data - x_min) / (x_max - x_min)
    weights_0_to_255_uint8 = (weights_0_to_1 * 255).astype(np.uint8)

    # 転置 (5, 5, 3, 32) -> (32, 5, 5, 3)
    image_data = np.transpose(weights_0_to_255_uint8, [3, 0, 1, 2])

    # 画像に1pxの枠線を追加し、1画像ずつリストに格納
    list = []
    for data in image_data:
        data = np.pad(data, pad_width=((1, 1), (1, 1), (0, 0)),
                  mode='constant', constant_values=0)
        list.append(data)

    # 64個のフィルターをグリッドに整形
    rows = None
    grid_size_height = 4
    grid_size_width = 8
    for index in range(grid_size_height):
        start = index * grid_size_width
        end = start + grid_size_width
        row = np.hstack(list[start:end])
        if rows is None:
            rows = row
        else:
            rows = np.vstack((rows, row))

    # bmpファイルとして出力
    file_path = os.path.join('./output_bmp', name) + '.bmp'
    with open(file_path, mode='wb') as fp:
        Image.fromarray(rows, 'RGB').save(fp, format='bmp')


def visualize_conv_1(name, weights_data):
    # 値を0-255に変換
    x_min = np.amin(weights_data)
    x_max = np.amax(weights_data)
    weights_0_to_1 = (weights_data - x_min) / (x_max - x_min)
    weights_0_to_255_uint8 = (weights_0_to_1 * 255).astype(np.uint8)

    # 転置 (50, 28, 28, 32) -> (50, 32, 28, 28)
    image_data = np.transpose(weights_0_to_255_uint8, [0, 3, 1, 2])

    # 1チャネルにreshapeする (50, 28, 28, 32) -> (50*32, 28, 28)
    image_data = np.reshape(image_data, (50*32, 28, 28))

    # 画像に1pxの枠線を追加し、1画像ずつリストに格納
    list = []
    for data in image_data:
        data = np.pad(data, pad_width=((1, 1), (1, 1)),
                  mode='constant', constant_values=0)
        list.append(data)

    # 50*32個のフィルターをグリッドに整形
    rows = None
    grid_size_height = 40
    grid_size_width = 40
    for index in range(grid_size_height):
        start = index * grid_size_width
        end = start + grid_size_width
        row = np.hstack(list[start:end])
        if rows is None:
            rows = row
        else:
            rows = np.vstack((rows, row))

    # bmpファイルとして出力
    file_path = os.path.join('./output_bmp', name) + '.bmp'
    with open(file_path, mode='wb') as fp:
        Image.fromarray(rows, 'L').save(fp, format='bmp')



def visualize_pool_1(name, weights_data):
    # 値を0-255に変換
    x_min = np.amin(weights_data)
    x_max = np.amax(weights_data)
    weights_0_to_1 = (weights_data - x_min) / (x_max - x_min)
    weights_0_to_255_uint8 = (weights_0_to_1 * 255).astype(np.uint8)

    # 転置 (50, 14, 14, 32) -> (50, 32, 14, 14)
    image_data = np.transpose(weights_0_to_255_uint8, [0, 3, 1, 2])

    # 1チャネルにreshapeする (50, 32, 14, 14) -> (50*32, 14, 14)
    image_data = np.reshape(image_data, (50*32, 14, 14))

    # 画像に1pxの枠線を追加し、1画像ずつリストに格納
    list = []
    for data in image_data:
        data = np.pad(data, pad_width=((1, 1), (1, 1)),
                  mode='constant', constant_values=0)
        list.append(data)

    # 50*32個のフィルターをグリッドに整形
    rows = None
    grid_size_height = 40
    grid_size_width = 40
    for index in range(grid_size_height):
        start = index * grid_size_width
        end = start + grid_size_width
        row = np.hstack(list[start:end])
        if rows is None:
            rows = row
        else:
            rows = np.vstack((rows, row))

    # bmpファイルとして出力
    file_path = os.path.join('./output_bmp', name) + '.bmp'
    with open(file_path, mode='wb') as fp:
        Image.fromarray(rows, 'L').save(fp, format='bmp')


def visualize_weights_2(name, weights_data):
    # 値を0-255に変換
    x_min = np.amin(weights_data)
    x_max = np.amax(weights_data)
    weights_0_to_1 = (weights_data - x_min) / (x_max - x_min)
    weights_0_to_255_uint8 = (weights_0_to_1 * 255).astype(np.uint8)

    # 転置 (5, 5, 32, 64) -> (64, 32, 5, 5)
    image_data = np.transpose(weights_0_to_255_uint8, [3, 2, 0, 1])

    # 1チャネルにreshapeする (64, 5, 5, 32) -> (64*32, 5, 5)
    image_data = np.reshape(image_data, (64*32, 5, 5))

    # 画像に1pxの枠線を追加し、1画像ずつリストに格納
    list = []
    for data in image_data:
        data = np.pad(data, pad_width=((1, 1), (1, 1)),
                  mode='constant', constant_values=0)
        list.append(data)

    # 64*32個のフィルターをグリッドに整形
    rows = None
    grid_size_height = 64
    grid_size_width = 32
    for index in range(grid_size_height):
        start = index * grid_size_width
        end = start + grid_size_width
        row = np.hstack(list[start:end])
        if rows is None:
            rows = row
        else:
            rows = np.vstack((rows, row))

    # bmpファイルとして出力
    file_path = os.path.join('./output_bmp', name) + '.bmp'
    with open(file_path, mode='wb') as fp:
        Image.fromarray(rows, 'L').save(fp, format='bmp')


def visualize_conv_2(name, weights_data):
    # 値を0-255に変換
    x_min = np.amin(weights_data)
    x_max = np.amax(weights_data)
    weights_0_to_1 = (weights_data - x_min) / (x_max - x_min)
    weights_0_to_255_uint8 = (weights_0_to_1 * 255).astype(np.uint8)

    # 転置 (50, 14, 14, 64) -> (50, 64, 14, 14)
    image_data = np.transpose(weights_0_to_255_uint8, [0, 3, 1, 2])

    # 1チャネルにreshapeする (50, 64, 14, 14) -> (50*64, 14, 14)
    image_data = np.reshape(image_data, (50*64, 14, 14))

    # 画像に1pxの枠線を追加し、1画像ずつリストに格納
    list = []
    for data in image_data:
        data = np.pad(data, pad_width=((1, 1), (1, 1)),
                  mode='constant', constant_values=0)
        list.append(data)

    # 50*64個のフィルターをグリッドに整形
    rows = None
    grid_size_height = 64
    grid_size_width = 50
    for index in range(grid_size_height):
        start = index * grid_size_width
        end = start + grid_size_width
        row = np.hstack(list[start:end])
        if rows is None:
            rows = row
        else:
            rows = np.vstack((rows, row))

    # bmpファイルとして出力
    file_path = os.path.join('./output_bmp', name) + '.bmp'
    with open(file_path, mode='wb') as fp:
        Image.fromarray(rows, 'L').save(fp, format='bmp')



def visualize_pool_2(name, weights_data):
    # 値を0-255に変換
    x_min = np.amin(weights_data)
    x_max = np.amax(weights_data)
    weights_0_to_1 = (weights_data - x_min) / (x_max - x_min)
    weights_0_to_255_uint8 = (weights_0_to_1 * 255).astype(np.uint8)

    # 転置 (50, 7, 7, 64) -> (50, 64, 7, 7)
    image_data = np.transpose(weights_0_to_255_uint8, [0, 3, 1, 2])

    # 1チャネルにreshapeする (50, 64, 7, 7) -> (50*64, 7, 7)
    image_data = np.reshape(image_data, (50*64, 7, 7))

    # 画像に1pxの枠線を追加し、1画像ずつリストに格納
    list = []
    for data in image_data:
        data = np.pad(data, pad_width=((1, 1), (1, 1)),
                  mode='constant', constant_values=0)
        list.append(data)

    # 50*64個のフィルターをグリッドに整形
    rows = None
    grid_size_height = 64
    grid_size_width = 50
    for index in range(grid_size_height):
        start = index * grid_size_width
        end = start + grid_size_width
        row = np.hstack(list[start:end])
        if rows is None:
            rows = row
        else:
            rows = np.vstack((rows, row))

    # bmpファイルとして出力
    file_path = os.path.join('./output_bmp', name) + '.bmp'
    with open(file_path, mode='wb') as fp:
        Image.fromarray(rows, 'L').save(fp, format='bmp')

if __name__ == '__main__':
    # コマンドライン引数を取得
    name = arg[1]
    weights_data = arg[2]

    if name == 'w1':
        visualize_weights_1(name, weights_data)
    elif name == 'c1':
        visualize_conv_1(name, weights_data)
    elif name == 'p1':
        visualize_pool_1(name, weights_data)
    elif name == 'w2':
        visualize_weights_2(name, weights_data)
    elif name == 'c2':
        visualize_conv_2(name, weights_data)
    elif name == 'p2':
        visualize_pool_2(name, weights_data)
    else:
        pass
