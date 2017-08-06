# Chainer Imagenet Tutorial


Datasetの準備

[Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)を使用

```
$ wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
$ tar xzvf 101_ObjectCategories.tar.gz
$ mv 101_ObjectCategories original
```

Datasetから不要カテゴリを削除

```
$ rm -rf ./original/BACKGROUND_Google
$ rm -rf ./original/Faces
$ rm -rf ./original/Faces_easy
```

データ準備

```
$ python make_data.py
$ python compute_mean.py train.txt
```

学習

```
$ python train_imagenet.py -a nin -E 500 -g 0 train.txt test.txt --test
```

推定

```
$ python predict_cmd.py ../target/sample1.jpg --initmodel ./result/model_iter_10000
```


compute_mean.py, nin.py from [chainer/chainer](https://github.com/chainer/chainer)
