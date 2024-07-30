# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 18:21:50 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255, x_test / 255

np2Neurons = 7
nConv = 3
nHidden = 3
np2ConvFeat = 2
log_lr = -3
drop_out = 0.2
kernel_shape = 4
max_tries = 30

best_validation_rate = 0
best_np2Neurons = np2Neurons
best_nConv = nConv
best_np2ConvFeat = np2ConvFeat
best_nHidden = nHidden
best_log_lr = log_lr
best_drop_out = drop_out
best_kernel_shape = kernel_shape

K = len(set(y_train.flatten()))

for trial in range(max_tries):
    if trial == max_tries - 1:
        np2Neurons = best_np2Neurons
        nConv = best_nConv
        np2ConvFeat = best_np2ConvFeat
        nHidden = best_nHidden
        log_lr = best_log_lr
        drop_out = best_drop_out
    
    opt = tf.keras.optimizers.Adam(10**log_lr)
    kernel = (kernel_shape, kernel_shape)

    print('Hidden Neurons:', 2**np2Neurons, 
          'Conv layers:', nConv, 
          'Conv features:', 2**np2ConvFeat, 
          'Hidden layers:', nHidden, 
          'Learning rate:', 10**log_lr,
          'drop out:', drop_out,
          'kernel shape:', kernel)        
    
    i = Input(shape = x_train[0].shape)
    x = Conv2D(2**np2ConvFeat, kernel, strides = 2, padding = 'same')(i)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    for l_conv in range(nConv):    
        x = Conv2D(2**(np2ConvFeat+l_conv+1), kernel, strides = 2, padding = 'same', activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
    x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    
    for l_hidden in range(nHidden):
        x = Dropout(drop_out)(x)
        x = Dense(2**np2Neurons)(x)
        x = Activation('relu')(x)
        
    x = Dropout(drop_out)(x)
    x = Dense(K, 'softmax')(x)
    
    model = Model(i, x)
    model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    r = model.fit(x = x_train, y = y_train, batch_size = 500, validation_data = (x_test, y_test), epochs = 20)
    
    validation_accuracy = r.history['val_accuracy'][-1]
    train_accuracy = r.history['accuracy'][-1]
    print('trial number:', trial + 1,'val accuracy', validation_accuracy, 'train accuracy', train_accuracy)

    
    if validation_accuracy > best_validation_rate and validation_accuracy / train_accuracy > 0.95:
        print("New best set found!")
        best_validation_rate = validation_accuracy
        best_np2Neurons = np2Neurons
        best_nConv = nConv
        best_np2ConvFeat = np2ConvFeat
        best_nHidden = nHidden
        best_log_lr = log_lr
        best_drop_out = drop_out
        best_kernel_shape = kernel_shape
                
    nHidden = best_nHidden + np.random.randint(-1, 2)
    nHidden = max(1, nHidden)
    nHidden = min(4, nHidden)
    nConv = best_nConv + np.random.randint(-1, 2)
    nConv = max(1, nConv)
    nConv = min(3, nConv)
    np2Neurons = best_np2Neurons + np.random.randint(-1, 2)
    np2Neurons = min(11, np2Neurons)
    np2ConvFeat = best_np2ConvFeat + np.random.randint(-1, 2)
    np2ConvFeat = min(4, np2ConvFeat)
    log_lr = best_log_lr + np.random.randint(-1, 2)
    drop_out = round(best_drop_out + np.random.randint(-1, 2)*0.1,1)
    drop_out = max(0, drop_out)
    kernel_shape = best_kernel_shape + np.random.randint(-1, 2)
    kernel_shape = max(0, kernel_shape)
    kernel_shape = min(6, kernel_shape)
    
print("best accuracy", best_validation_rate)
print('best settings:')
print("best_np2Neurons:", best_np2Neurons)
print("best_nConv", best_nConv)
print("best_np2ConvFeat:", best_np2ConvFeat)
print("best_nHidden", best_nHidden)
print("best_log_lr:", best_log_lr)
print("best_drop_out", best_drop_out)
print("best_best_kernel_shape", best_kernel_shape)

plt.plot(r.history['loss'], label = 'loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history["val_accuracy"], label='val_accuracy')
plt.legend()
plt.show()

print(model.evaluate(x_test, y_test))

def plot_confusion_matrix(cm, 
                          classes, 
                          normalize=False, 
                          title='Confusion_matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))
