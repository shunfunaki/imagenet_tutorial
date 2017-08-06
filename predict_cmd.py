# coding: utf-8
from __future__ import print_function
from flask import Flask, request, jsonify
import argparse
import cv2
import numpy as np
import chainer
import nin

WIDTH = 256                         # リサイズ後の幅
HEIGHT = 256                        # リサイズ後の高さ
LIMIT = 20                          # 出力上限

model = nin.NIN()

parser = argparse.ArgumentParser()
parser.add_argument('image', help='Path to inspection image file')
parser.add_argument('--initmodel', type=str, default='',
                    help='Initialize the model from given file')
args = parser.parse_args()


def preprocess(path, crop_size, mean):
    image = np.asarray(Image.open(path))
    # リサイズ
    image = cv2.resize(image, (WIDTH, HEIGHT))

    # (height, width, channel) -> (channel, height, width) に変換
    image = image.transpose(2, 0, 1)

    _, h, w = image.shape

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    bottom = top + crop_size
    right = left + crop_size

    image = image[:, top:bottom, left:right].astype(np.float32)
    image -= mean[:, top:bottom, left:right]
    image /= 255

    return image

# 平均画像読み込みとモデルの復元
mean = np.load('mean.npy')
chainer.serializers.load_npz(args.initmodel, model)

# 入力画像を変換
img = preprocess(args.image, model.insize, mean)
x = np.ndarray((1, 3, model.insize, model.insize), dtype=np.float32)
x[0]=img
x = chainer.Variable(np.asarray(x))

# 分類実行
score = model.predict(x)

# 結果出力
categories = np.loadtxt("labels.txt", str, delimiter="\t")
prediction = zip(score.data[0].tolist(), categories)
prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
for rank, (score, name) in enumerate(prediction[:LIMIT], start=1):
    print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
