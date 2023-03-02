import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization, Conv2D, GlobalMaxPooling2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time

# d6 classification conditions
def work_condition_data_6class(frame_size, step, data_size, path_list):
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
# s8 classification conditions
def work_condition_data_8class(frame_size, step, data_size, path_list):
    spur1, spur2, spur3, spur4, spur5, spur6,spur7, spur8 = [], [], [], [], [], [], [], []
    work1 = np.loadtxt('./data/' + path_list[0])[:, 1]
    work2 = np.loadtxt('./data/' + path_list[1])[:, 1]
    work3 = np.loadtxt('./data/' + path_list[2])[:, 1]
    work4 = np.loadtxt('./data/' + path_list[3])[:, 1]
    work5 = np.loadtxt('./data/' + path_list[4])[:, 1]
    work6 = np.loadtxt('./data/' + path_list[5])[:, 1]
    work7 = np.loadtxt('./data/' + path_list[6])[:, 1]
    work8 = np.loadtxt('./data/' + path_list[7])[:, 1]

    for i in range(data_size):
        spur1.append(work1[i * step: i * step + frame_size].tolist())
        spur2.append(work2[i * step: i * step + frame_size].tolist())
        spur3.append(work3[i * step: i * step + frame_size].tolist())
        spur4.append(work4[i * step: i * step + frame_size].tolist())
        spur5.append(work5[i * step: i * step + frame_size].tolist())
        spur6.append(work6[i * step: i * step + frame_size].tolist())
        spur7.append(work7[i * step: i * step + frame_size].tolist())
        spur8.append(work8[i * step: i * step + frame_size].tolist())
    data = np.concatenate((np.array(spur1), np.array(spur2), np.array(spur3), np.array(spur4), np.array(spur5), np.array(spur6), np.array(spur7), np.array(spur8)), axis=0)
    labels = np.zeros((data_size * 8, 8))
    labels[:data_size, 0] = 1
    labels[data_size:2 * data_size, 1] = 1
    labels[2 * data_size:3 * data_size, 2] = 1
    labels[3 * data_size:4 * data_size, 3] = 1
    labels[4 * data_size:5 * data_size, 4] = 1
    labels[5 * data_size:6 * data_size, 5] = 1
    labels[6 * data_size:7 * data_size, 6] = 1
    labels[7 * data_size:, 7] = 1

    return data, labels
# CNN Base Model
def cnn_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split):
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
    #model.summary()

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

    return model, history
# CNN migration model
def cnn_transfer_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split):
    base_model = load_model('cnn_CWRU_model_50.h5')
    #base_model = load_model('cnn_gearbox_model_50.h5')
    x = base_model.get_layer('fc_1').output
    x = Dropout(0.5, name='dropout_3')(x)
    output = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    '''
    # 只fine-turning最后一层
    for layer in model.layers[:-3]:
        layer.trainable = False
    for layer in model.layers[-3:]:
        layer.trainable = True
    '''
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
    return model, history
# MLP Base Model
def mlp_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split):
    la = 0.001
    model = Sequential()
    model.add(Dense(512, input_dim=frame_size, activation='relu', name='fc', kernel_regularizer=l2(la)))
    model.add(Dense(256, activation='relu', name='dense', kernel_regularizer=l2(la)))
    model.add(Dense(class_num, activation='softmax', name='output_layer', kernel_regularizer=l2(la)))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])

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

    return model, history
# MLP Transfer Model
def mlp_transfer_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split):
    #base_model = load_model('mlp_CWRU_model_50.h5')
    base_model = load_model('mlp_gearbox_model_50.h5')
    x = base_model.get_layer('dense1').output
    output = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    ''''''
    # Only fine-turning the last layer
    for layer in model.layers[:-3]:
        layer.trainable = False
    for layer in model.layers[-3:]:
        layer.trainable = True

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    #model.summary()

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
    return model, history




work_condition_1_path_list = [
    'helical 1/helical 1_30hz_High_1.txt', 'helical 2/helical 2_30hz_High_1.txt', 'helical 3/helical 3_30hz_High_1.txt', 'helical 4/helical 4_30hz_High_1.txt',
    'helical 5/helical 5_30hz_High_1.txt', 'helical 6/helical 6_30hz_High_1.txt',
]
work_condition_2_path_list = [
    'helical 1/helical 1_35hz_Low_1.txt', 'helical 2/helical 2_35hz_Low_1.txt', 'helical 3/helical 3_35hz_Low_1.txt', 'helical 4/helical 4_35hz_Low_1.txt',
    'helical 5/helical 5_35hz_Low_1.txt', 'helical 6/helical 6_35hz_Low_1.txt',
]
#work_condition_2_path_list = [
#    'helical 1/helical 1_30hz_Low_2.txt', 'helical 2/helical 2_30hz_Low_2.txt', 'helical 3/helical 3_30hz_Low_2.txt', 'helical 4/helical 4_30hz_Low_2.txt',
#    'helical 5/helical 5_30hz_Low_2.txt', 'helical 6/helical 6_30hz_Low_2.txt',
#]
work_condition_3_path_list = [
    'spur 1/spur 1_30hz_High_1.txt', 'spur 2/spur 2_30hz_High_1.txt', 'spur 3/spur 3_30hz_High_1.txt', 'spur 4/spur 4_30hz_High_1.txt', 'spur 5/spur 5_30hz_High_1.txt',
    'spur 6/spur 6_30hz_High_1.txt', 'spur 7/spur 7_30hz_High_1.txt', 'spur 8/spur 8_30hz_High_1.txt',
]
work_condition_4_path_list = [
    'spur 1/spur 1_30hz_Low_1.txt', 'spur 2/spur 2_30hz_Low_1.txt', 'spur 3/spur 3_30hz_Low_1.txt', 'spur 4/spur 4_30hz_Low_1.txt', 'spur 5/spur 5_30hz_Low_1.txt',
    'spur 6/spur 6_30hz_Low_1.txt', 'spur 7/spur 7_30hz_Low_1.txt', 'spur 8/spur 8_30hz_Low_1.txt',
]
work_condition_5_path_list = [
    'spur 1/spur 1_50hz_High_1.txt', 'spur 2/spur 2_50hz_High_1.txt', 'spur 3/spur 3_50hz_High_1.txt', 'spur 4/spur 4_50hz_High_1.txt', 'spur 5/spur 5_50hz_High_1.txt',
    'spur 6/spur 6_50hz_High_1.txt', 'spur 7/spur 7_50hz_High_1.txt', 'spur 8/spur 8_50hz_High_1.txt',
]
work_condition_6_path_list = [
    'spur 1/spur 1_50hz_Low_1.txt', 'spur 2/spur 2_50hz_Low_1.txt', 'spur 3/spur 3_50hz_Low_1.txt', 'spur 4/spur 4_50hz_Low_1.txt', 'spur 5/spur 5_50hz_Low_1.txt',
    'spur 6/spur 6_50hz_Low_1.txt', 'spur 7/spur 7_50hz_Low_1.txt', 'spur 8/spur 8_50hz_Low_1.txt',
]


'''
work_condition_3_path_list = [
    'helical 1/helical 1_50hz_High_1.txt', 'helical 2/helical 2_50hz_High_1.txt', 'helical 3/helical 3_50hz_High_1.txt', 'helical 4/helical 4_50hz_High_1.txt',
    'helical 5/helical 5_50hz_High_1.txt', 'helical 6/helical 6_50hz_High_1.txt',
]
work_condition_4_path_list = [
    'helical 1/helical 1_50hz_Low_1.txt', 'helical 2/helical 2_50hz_Low_1.txt', 'helical 3/helical 3_50hz_Low_1.txt', 'helical 4/helical 4_50hz_Low_1.txt',
    'helical 5/helical 5_50hz_Low_1.txt', 'helical 6/helical 6_50hz_Low_1.txt',
]
'''



frame_size = 1000
step = 50
data_size = 400
class_num = 6
batch_size = 16
split = 0.2
#  Read data
data, labels = work_condition_data_6class(frame_size, step, data_size, work_condition_2_path_list)
#data, labels = work_condition_data_8class(frame_size, step, data_size, work_condition_5_path_list)
print("work condition 1 total data shape:", data.shape)
print("work condition 1 total labels shape:", labels.shape)


plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 20
plt.subplot(611)
plt.plot(data[0, :], linewidth=1)
plt.ylim((-0.1, 0.1))
plt.title('Label 1')
plt.subplot(612)
plt.plot(data[500, :], linewidth=1)
plt.ylim((-0.1, 0.1))
plt.title('Label 2')
plt.subplot(613)
plt.plot(data[900, :], linewidth=1)
plt.ylim((-0.1, 0.1))
plt.title('Label 3')
plt.subplot(614)
plt.plot(data[1500, :], linewidth=1)
plt.ylim((-0.1, 0.1))
plt.title('Label 4')
plt.subplot(615)
plt.plot(data[1700, :], linewidth=1)
plt.ylim((-0.1, 0.1))
plt.title('Label 5')
plt.subplot(616)
plt.plot(data[2100, :], linewidth=1)
plt.ylim((-0.1, 0.1))
plt.title('Label 6')
plt.show()



# 50% cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=4)
#training
n = 0
train_acc_list, val_acc_list, test_acc_list = [], [], []
time1 = time.clock()
for train_index, test_index in kf.split(data):
    time3 = time.clock()
    print(str(n+1) + "**********************************************************************")
    x_train, y_train = data[train_index], labels[train_index]
    x_test, y_test = data[test_index], labels[test_index]
    print("the number of every class:", np.bincount(np.transpose(np.nonzero(y_train))[:, 1]))
    # upset
    index = [i for i in range(x_train.shape[0])]
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]
    # standardization
    x_train = StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)
    # reshape
    x_train = x_train.reshape((-1, frame_size, 1))
    x_test = x_test.reshape((-1, frame_size, 1))
    # Read CNN model
    model, history = cnn_transfer_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split)
    #model, history = cnn_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split)
    # Read MLP model
    #model, history = mlp_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split)
    #model, history = mlp_transfer_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split)

    # Output training history information
    print("acc:", history.history['binary_accuracy'])
    print("val_acc:", history.history['val_binary_accuracy'])

    # prediction
    predict = model.predict(x_test, batch_size=1, verbose=1)
    correct_num = 0
    confuse_mat = np.array(np.zeros((class_num, class_num)))    # 混淆矩阵
    for i in range(predict.shape[0]):
        indexpr = np.where(predict[i, :] == np.max(predict[i, :]))
        indextest = np.where(y_test[i, :] == np.max(y_test[i, :]))
        if indexpr == indextest:
            correct_num = correct_num + 1
        confuse_mat[indexpr[0][0]-1, indextest[0][0]-1] += 1
    acc = correct_num/predict.shape[0]
    print("acc:", acc)
    print("confuse mat:", confuse_mat)
    n = n + 1

    train_acc_list.append(history.history['binary_accuracy'])
    val_acc_list.append(history.history['val_binary_accuracy'])
    test_acc_list.append(acc)
    time4 = time.clock()
    print("running time:", str(time4 - time3))
time2 = time.clock()
#print("mean train acc:", np.mean(np.array(train_acc_list), axis=0))
#print("mean val acc:", np.mean(np.array(val_acc_list), axis=0))
print("train acc list:", train_acc_list)
print("val acc list:", val_acc_list)
print("test acc list:", test_acc_list)
print("mean test acc:", np.mean(test_acc_list))
print("total running time:", str(time2 - time1))



