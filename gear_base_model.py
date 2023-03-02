import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization, Conv2D, GlobalMaxPooling2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model
import matplotlib.pyplot as plt
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import time

# Working condition data of training base model
def work_data(frame_size, step, data_size, path_list):
    helical1, helical2, helical3, helical4, helical5, helical6, = [], [], [], [], [], []
    work1 = np.loadtxt('./data/' + path_list[0])[:, 1]
    work2 = np.loadtxt('./data/' + path_list[1])[:, 1]
    work3 = np.loadtxt('./data/' + path_list[2])[:, 1]
    work4 = np.loadtxt('./data/' + path_list[3])[:, 1]
    work5 = np.loadtxt('./data/' + path_list[4])[:, 1]
    work6 = np.loadtxt('./data/' + path_list[5])[:, 1]

    for i in range(data_size):
        helical1.append(work1[i * step: i * step + frame_size].tolist())
        helical2.append(work2[i * step: i * step + frame_size].tolist())
        helical3.append(work3[i * step: i * step + frame_size].tolist())
        helical4.append(work4[i * step: i * step + frame_size].tolist())
        helical5.append(work5[i * step: i * step + frame_size].tolist())
        helical6.append(work6[i * step: i * step + frame_size].tolist())
    data = np.concatenate((np.array(helical1), np.array(helical2), np.array(helical3), np.array(helical4), np.array(helical5), np.array(helical6)), axis=0)
    labels = np.zeros((data_size * 6, 6))
    labels[:data_size, 0] = 1
    labels[data_size:2 * data_size, 1] = 1
    labels[2 * data_size:3 * data_size, 2] = 1
    labels[3 * data_size:4 * data_size, 3] = 1
    labels[4 * data_size:5 * data_size, 4] = 1
    labels[5 * data_size:, 5] = 1

    return data, labels
# CNN Base Model
def cnn_base_model(x_train, y_train, x_val, y_val, frame_size, class_num, batch_size, save, split):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=(15), strides=(2), activation='relu', input_shape=(frame_size, 1), padding='same', name='conv_1'))
    model.add(MaxPooling1D(name='max_pooling'))
    model.add(Dropout(0.5, name='dropout_1'))
    model.add(Conv1D(filters=32, kernel_size=(15), strides=(2), activation='relu', padding='same', name='conv_2'))
    model.add(MaxPooling1D(name='max_pooling_1'))
    model.add(Dropout(0.5, name='dropout_2'))
    model.add(Conv1D(filters=64, kernel_size=(15), strides=(2), activation='relu', padding='same', name='conv_3'))
    model.add(MaxPooling1D(name='max_pooling_2'))
    #model.add(Dropout(0.5, name='dropout_3'))
    #model.add(GlobalMaxPooling1D(name='global_max_pooling'))
    model.add(Flatten())

    model.add(Dense(512, activation='relu', name='fc_1', kernel_regularizer=l2(0.00001)))#kernel_regularizer=l2(0.00005),
    model.add(Dropout(0.5, name='dropout_4'))
    model.add(Dense(class_num, activation='softmax', name='output'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    model.summary()

    early = EarlyStopping(monitor='val_binary_accuracy', min_delta=0, patience=20, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.1, patience=9, verbose=1, epsilon=1e-4, mode='min')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=50,
                        verbose=2,
                        callbacks=[reduce_lr_loss, early],
                        validation_data=(x_val, y_val),
                        #validation_split=0.1
                        )
    if save:
        model.save('cnn_gearbox_model_50.h5')
        #model.save('cnn_gearbox_model_16filtersize.h5')
    return model, history
# MLP Base Modsel
def mlp_base_model(x_train, y_train, x_val, y_val, frame_size, class_num, batch_size, save, split):
    la = 0.001
    model = Sequential()
    model.add(Dense(512, input_dim=frame_size, activation='relu', name='fc', kernel_regularizer=l2(la)))
    model.add(Dense(256, activation='relu', name='dense1', kernel_regularizer=l2(la)))
    model.add(Dense(class_num, activation='softmax', name='output_layer', kernel_regularizer=l2(la)))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    model.summary()

    early = EarlyStopping(monitor='val_binary_accuracy', min_delta=0, patience=20, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.1, patience=9, verbose=1, epsilon=1e-4, mode='min')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=50,
                        verbose=2,
                        callbacks=[reduce_lr_loss, early],
                        validation_data=(x_val, y_val),
                        #validation_split=split
                        )
    if save:
        model.save('mlp_gearbox_model_50.h5')
        #model.save('mlp_gearbox_model_new.h5')
        
        
    return model, history


data_path = [
    'helical 1/helical 1_35hz_High_1.txt', 'helical 2/helical 2_35hz_High_1.txt', 'helical 3/helical 3_35hz_High_1.txt',
    'helical 4/helical 4_35hz_High_1.txt', 'helical 5/helical 5_35hz_High_1.txt', 'helical 6/helical 6_35hz_High_1.txt'
]

frame_size = 1000
step = 50
data_size = 2000
class_num = 6
batch_size = 128
save = False
split = 0.2
# Read data
train_data, labels = work_data(frame_size, step, data_size, data_path)
print("work condition total data shape:", train_data.shape)
print("work condition total labels shape:", labels.shape)
# Divide training set and test set
x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
# standardization
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
x_val = StandardScaler().fit_transform(x_val)
# reshape
x_train = x_train.reshape((-1, frame_size, 1))
x_test = x_test.reshape((-1, frame_size, 1))
x_val = x_val.reshape((-1, frame_size, 1))
print("work condition 1 x_train shape:", x_train.shape)
print("work condition 1 x_test shape:", x_test.shape)

time1 = time.clock()
# Train CNN model
model, history = cnn_base_model(x_train, y_train, x_val, y_val, frame_size, class_num, batch_size, save, split)
# Train MLP model
x_train = x_train.reshape((-1, frame_size))
x_test = x_test.reshape((-1, frame_size))
x_val = x_val.reshape((-1, frame_size))
model, history = mlp_base_model(x_train, y_train, x_val, y_val, frame_size, class_num, batch_size, save, split)

time2 = time.clock()
# Output training history information
print("acc:", history.history['binary_accuracy'])
print("val_acc:", history.history['val_binary_accuracy'])

# prediction
predict = model.predict(x_test, batch_size=1, verbose=0)
correct_num = 0
confuse_mat = np.array(np.zeros((class_num, class_num)))    # 混淆矩阵
for i in range(predict.shape[0]):
    indexpr = np.where(predict[i, :] == np.max(predict[i, :]))
    indextest = np.where(y_test[i, :] == np.max(y_test[i, :]))
    if indexpr == indextest:
        correct_num = correct_num + 1
    confuse_mat[indexpr[0][0]-1, indextest[0][0]-1] += 1
acc = correct_num/predict.shape[0]
print("test acc:", acc)
print("training time:", str(time2 - time1))
print("confuse mat:", confuse_mat)










