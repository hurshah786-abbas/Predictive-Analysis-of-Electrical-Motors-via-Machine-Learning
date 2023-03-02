import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization, Conv2D, GlobalMaxPooling2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.optimizers import Adam # - Works

#Working condition data of training base model
def work_data(frame_size, step, data_size, path_list):
    ball_data, inner_data, outer_data, normal_data = [], [], [], []
    ball_fault = np.loadtxt('./data/' + path_list[0])
    inner_race_fault = np.loadtxt('./data/' + path_list[1])
    outer_race_center_fault = np.loadtxt('./data/' + path_list[2])
    normal = np.loadtxt('./data/' + path_list[3])

    for i in range(data_size):
        ball_data.append(ball_fault[i * step: i * step + frame_size].tolist())
        inner_data.append(inner_race_fault[i * step: i * step + frame_size].tolist())
        outer_data.append(outer_race_center_fault[i * step: i * step + frame_size].tolist())
        normal_data.append(normal[i * step: i * step + frame_size].tolist())
    data = np.concatenate((np.array(ball_data), np.array(inner_data), np.array(outer_data), np.array(normal_data)), axis=0)
    labels = np.zeros((data_size * 4, 4))
    labels[:data_size, 0] = 1
    labels[data_size:2 * data_size, 1] = 1
    labels[2 * data_size:3 * data_size, 2] = 1
    labels[3 * data_size:, 3] = 1

    return data, labels
# CNN-based model
def cnn_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, save, split):
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

    model.add(Dense(512, activation='relu', name='fc_1'))
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
                        #validation_data=(x_test, y_test),
                        validation_split=split
                        )
    if save:
        model.save('cnn_CWRU_model_50.h5')
       # model.save('cnn_CWRU_model_16filtersize.h5')

    return model, history
#  MLP based model
def mlp_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, save, split):
    la = 0.001
    model = Sequential()
    model.add(Dense(512, input_dim=frame_size, activation='relu', name='fc', kernel_regularizer=l2(la)))
    model.add(Dense(256, activation='relu', name='dense', kernel_regularizer=l2(la)))
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
                        #validation_data=(x_test, y_test),
                        validation_split=split
                        )
    if save:
        #model.save('mlp_CWRU_model_new.h5')
        model.save('mlp_CWRU_model_50.h5')

    return model, history


k48_drive_4class_path_list = [
    '48k_Drive_End_Bearing_Fault_Data/Ball_Fault/123_1772.csv',
    '48k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/110_1772.csv',
    '48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/202_1772.csv',
    '48k_Normal_Baseline_Data/98_1772.csv'
]
frame_size = 1000
step = 50
data_size = 2000
class_num = 4
batch_size = 128
save = True
split = 0.2

#Read data
train_data, labels = work_data(frame_size, step, data_size, k48_drive_4class_path_list)
print("work condition total data shape:", train_data.shape)
print("work condition total labels shape:", labels.shape)

# Divide training set and test set
x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2)
# Standardization
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
# reshape
x_train = x_train.reshape((-1, frame_size, 1))
x_test = x_test.reshape((-1, frame_size, 1))
print("work condition 1 x_train shape:", x_train.shape)
print("work condition 1 x_test shape:", x_test.shape)

time1 = time.clock()
# Train CNN model
model, history = cnn_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, save, split)
# Train MLP model
x_train = x_train.reshape((-1, frame_size))
x_test = x_test.reshape((-1, frame_size))
model, history = mlp_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, save, split)
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
print("running time:", str(time2 - time1))
print("confuse mat:", confuse_mat)
















