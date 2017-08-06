# coding: utf-8
import os
import commands
import shutil
import re
import random
import cv2
import numpy as np


SRC_BASE_PATH = './original'            # ダウンロードした画像が格納されているベースディレクトリ

WIDTH = 256                             # リサイズ後の幅
HEIGHT = 256                            # リサイズ後の高さ
DST_BASE_PATH = './resized'             # リサイズ後の画像を格納するベースディレクトリ
LABELS_PATH = 'labels.txt'              # ラベルデータの出力先
TRAINVAL_PATH = 'train.txt'             # 学習データの出力先
TESTVAL_PATH = 'test.txt'               # 検証データの出力先
VAL_RATE = 0.25                         # 検証データの割合


if __name__ == '__main__':
    # ディレクトリからlabel作成
    dirs = commands.getoutput("ls " + SRC_BASE_PATH)
    labels = dirs.splitlines()

    # 画像の格納先を初期化
    if os.path.exists(DST_BASE_PATH):
        shutil.rmtree(DST_BASE_PATH)
    os.mkdir(DST_BASE_PATH)

    # 出力ファイルの削除
    if os.path.exists(LABELS_PATH):
        os.remove(LABELS_PATH)
    if os.path.exists(TRAINVAL_PATH):
        os.remove(TRAINVAL_PATH)
    if os.path.exists(TESTVAL_PATH):
        os.remove(TESTVAL_PATH)

    train_dataset = []
    val_dataset = []
    label_dataset = []
    idx = 0
    label_num = 0

    for label in labels:
        class_dir_path = os.path.join(SRC_BASE_PATH, label)

        files = [
            file for file in os.listdir(class_dir_path)
            if re.search(r'\.(jpe?g)$', file, re.IGNORECASE)
        ]

        rename_files = []

        # リサイズして画像ファイル出力
        for file in files:
            src_path = os.path.join(class_dir_path, file)
            image = cv2.imread(src_path)
            resized_image = cv2.resize(image, (WIDTH, HEIGHT))
            cv2.imwrite(os.path.join(DST_BASE_PATH, "image%07d" % idx + ".jpg"), resized_image)
            rename_files.append(os.path.join(DST_BASE_PATH, "image%07d" % idx + ".jpg"))
            idx += 1

        # 学習・検証データを作成
        bound = int(len(rename_files) * (1 - VAL_RATE))
        random.shuffle(rename_files)
        train_files = rename_files[:bound]
        val_files = rename_files[bound:]

        train_dataset.extend([(os.path.abspath(file), "%d" % label_num) for file in train_files])
        val_dataset.extend([(os.path.abspath(file), "%d" % label_num) for file in val_files])

        label_dataset.append(label)
        label_num += 1


    # ラベルデータを出力
    with open(LABELS_PATH, 'w') as f:
        for d in label_dataset:
            f.write(d + '\n')

    # 学習データを出力
    with open(TRAINVAL_PATH, 'w') as f:
        for d in train_dataset:
            f.write(' '.join(d) + '\n')

    # 検証データを出力
    with open(TESTVAL_PATH, 'w') as f:
        for d in val_dataset:
            f.write(' '.join(d) + '\n')
