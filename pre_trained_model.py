# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:32:31 2018

@author: vipli

Pre-trained model 
"""
import numpy as np
import os
from tqdm import tqdm_notebook
from random import shuffle
import shutil
from matplotlib import pyplot as plt
def organize_datasets(path_to_data, n=4000, ratio=0.2):
    files = os.listdir(path_to_data)
    files = [os.path.join(path_to_data, f) for f in files]
    shuffle(files)
    files = files[:n]

    n = int(len(files) * ratio)
    val, train = files[:n], files[n:]

    shutil.rmtree('./data/')
    print('/data/ removed')

    for c in ['dogs', 'cats']:
        os.makedirs('./data/train/{0}/'.format(c))
        os.makedirs('./data/validation/{0}/'.format(c))

    print('folders created !')

    for t in tqdm_notebook(train):
        if 'cat' in t:
            shutil.copy2(t, os.path.join('.', 'data', 'train', 'cats'))
        else:
            shutil.copy2(t, os.path.join('.', 'data', 'train', 'dogs'))

    for v in tqdm_notebook(val):
        if 'cat' in v:
            shutil.copy2(v, os.path.join('.', 'data', 'validation', 'cats'))
        else:
            shutil.copy2(v, os.path.join('.', 'data', 'validation', 'dogs'))

    print('Data copied!')

batch_size = 32
n = 1000
ratio = 0.2
#organize_datasets(path_to_data = 'D:/ML Data/Dogs vs. Cats/train/',n=n,ratio=ratio)

# 使用预训练网络VGG16来训练样本图片，获得图片的特征表示
# 再把特征输入自己定义的神经网络中进行分类
from keras.preprocessing.image import ImageDataGenerator
from keras import applications

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import Callback
from keras_tqdm import TQDMNotebookCallback


model = applications.VGG16(include_top=False, weights='imagenet') # 加载VGG16模型
datagen = ImageDataGenerator(rescale = 1. /255)
train_generator = datagen.flow_from_directory('./data/train/',target_size = (150,150),
                                              batch_size = batch_size,
                                              class_mode = None,
                                              shuffle = False)
bottleneck_features_train = model.predict_generator(train_generator, int(n * (1 - ratio)) // batch_size)
print(bottleneck_features_train.shape)
# 将预处理过的特征数据保存
np.save('./features/bottleneck_feature_train.npy', bottleneck_features_train)

validation_generator = datagen.flow_from_directory('./data/validation/',
                                        target_size=(150, 150),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)
bottleneck_features_validation = model.predict_generator(validation_generator, int(n * ratio) // batch_size,)
print(bottleneck_features_validation.shape)
np.save('./features/bottleneck_feature_validation.npy', bottleneck_features_validation)
print('pre-trained is Done')

# 加载预训练过程生成的特征数据，再定义全连接网络进行分类训练
train_data = np.load('./features/bottleneck_feature_train.npy')
validation_data =np.load('./features/bottleneck_feature_validation.npy')
train_labels = np.array([0] * (int(n*(1-ratio)) // 2) + [1] * (int(n*(1-ratio)) // 2))
validation_labels =  np.array([0] * (int(n*ratio) // 2) + [1] * (int(n*ratio) // 2))


model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

history = LossHistory()

fitted_model = model.fit(train_data, train_labels,epochs = 15,
                         batch_size = batch_size,
                         validation_data = (validation_data,validation_labels[:validation_data.shape[0]]),
                         verbose = 0, callbacks = [TQDMNotebookCallback(leave_inner=True,leave_outer=False),history])
# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
print('the training is done')


fig = plt.figure(figsize = (15,5))
plt.plot(fitted_model.history['loss'], 'g', label='train loss')
plt.plot(fitted_model.history['val_loss'], 'r', label='val loss')
plt.grid(True)
plt.title('Training loss vs. Validation loss - VGG16')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()

fig = plt.figure(figsize = (15,5))
plt.plot(fitted_model.history['acc'], 'g', label='accuracy on train set')
plt.plot(fitted_model.history['val_acc'], 'r', label = 'accuracy on validation set')
plt.grid(True)
plt.title('Train Accuracy vs. Validation Accuracy - VGG16')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()