#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
from tqdm import tqdm_notebook
from random import shuffle
import shutil

def organize_datasets(path_to_data, n=4000, ratio=0.2):
    """
    重新组织数据，将测试数据集分成train 和 validation 两个子数据集，在每个子数据集下
    再分成cats 和 dogs两个子数据集
    这样的目的是适应keras的应用程序编程接口(API)
    """
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

    print('data copied!')

n = 500
ratio = 0.2
#organize_datasets(path_to_data = 'D:/ML Data/Dogs vs. Cats/train/',n=n,ratio=ratio)

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMNotebookCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import Callback

batch_size = 32
train_datagen = ImageDataGenerator(rescale=1/255.,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True
                                    )
val_datagen = ImageDataGenerator(rescale=1/255.)


train_generator = train_datagen.flow_from_directory(
        './data/train/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')# 事先已经将训练数据分类到cats和dogs两个子文件夹中,这里就体现了keras的API
# class_mode 参数决定是否返回标签以及标签数组的形式。其值为'categorical','binary','sparse'或None之一
# 默认为'categorical',返回2D的one-hot编码标签
# 在这里,会返回关于猫狗分类的one-hot编码，如果是猫，则对应[[1.,0.]];如果是狗，则对应[[0.,1.]]
validation_generator = val_datagen.flow_from_directory(
        './data/validation/',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), padding='same', activation='relu'))
#  Output Shape(None,150,150,32) 模型参数： (3*3*3+1)*32=896
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# 模型参数：(3*3*32+1)*32=9248
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# 模型参数：(3*3*32+1)*64 = 18496
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

epochs = 50
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
# loss 定义目标损失函数，metrics定义性能评估函数
# 两者相似，但区别是性能评估的结果不会用于训练模型，模型是通过损失函数的误差来进行训练的
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

## Callback for early stopping the training
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

fitted_model = model.fit_generator(
        train_generator,
        steps_per_epoch= int(n * (1-ratio)) // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps= int(n * ratio) // batch_size,
        callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=True), early_stopping, history],
        verbose=0)
# model.fit(), model.fit_generator()返回一个Historty的对象，History.history属性记录了损失函数和其他指标的数值
# 随 epoch 变化的情况

# 预测结果可视化
fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['loss'], 'g', label="train losses")
plt.plot(fitted_model.history['val_loss'], 'r', label="val losses")
plt.grid(True)
plt.title('Training loss vs. Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['acc'], 'g', label="accuracy on train set")
plt.plot(fitted_model.history['val_acc'], 'r', label="accuracy on validation set")
plt.grid(True)
plt.title('Training Accuracy vs. Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
