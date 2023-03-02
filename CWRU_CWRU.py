import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization, Conv2D, GlobalMaxPooling2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.models import Model
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time

# 4 Classification conditions
def work_condition_data_4class(frame_size, step, data_size, path_list):
    ball, inner, outer_cen, normal = [], [], [], []
    work1 = np.loadtxt('./data/' + path_list[0])
    work2 = np.loadtxt('./data/' + path_list[1])
    work3 = np.loadtxt('./data/' + path_list[2])
    work4 = np.loadtxt('./data/' + path_list[3])
    b = path_list[-1]
    for i in range(data_size):
        ball.append(work1[i * step: i * step + frame_size].tolist())
        inner.append(work2[i * step: i * step + frame_size].tolist())
        outer_cen.append(work3[i * step: i * step + frame_size].tolist())
        normal.append(work4[b + i * step: b + i * step + frame_size].tolist())
    data = np.concatenate((np.array(ball), np.array(inner), np.array(outer_cen), np.array(normal)), axis=0)
    labels = np.zeros((data_size * 4, 4))
    labels[:data_size, 0] = 1
    labels[data_size:2 * data_size, 1] = 1
    labels[2 * data_size:3 * data_size, 2] = 1
    labels[3 * data_size:, 3] = 1

    return data, labels
# 6 Classification conditions
def work_condition_data_6class(frame_size, step, data_size, path_list):
    ball, inner, outer_cen, outer_opp, outer_orth, normal = [], [], [], [], [], []
    work1 = np.loadtxt('./data/' + path_list[0])
    work2 = np.loadtxt('./data/' + path_list[1])
    work3 = np.loadtxt('./data/' + path_list[2])
    work4 = np.loadtxt('./data/' + path_list[3])
    work5 = np.loadtxt('./data/' + path_list[4])
    work6 = np.loadtxt('./data/' + path_list[5])
    b = path_list[-1]
    for i in range(data_size):
        ball.append(work1[i * step: i * step + frame_size].tolist())
        inner.append(work2[i * step: i * step + frame_size].tolist())
        outer_cen.append(work3[i * step: i * step + frame_size].tolist())
        outer_opp.append(work4[i * step: i * step + frame_size].tolist())
        outer_orth.append(work5[i * step: i * step + frame_size].tolist())
        normal.append(work6[b + i * step: b + i * step + frame_size].tolist())
    data = np.concatenate((np.array(ball), np.array(inner), np.array(outer_cen), np.array(outer_opp), np.array(outer_orth), np.array(normal)), axis=0)
    labels = np.zeros((data_size * 6, 6))
    labels[:data_size, 0] = 1
    labels[data_size:2 * data_size, 1] = 1
    labels[2 * data_size:3 * data_size, 2] = 1
    labels[3 * data_size:4 * data_size, 3] = 1
    labels[4 * data_size:5 * data_size, 4] = 1
    labels[5 * data_size:, 5] = 1

    return data, labels
# CNN Model
def cnn_base_model(x_train, y_train, x_val, y_val, frame_size, class_num, batch_size, split):
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
    sgd = SGD(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    #model.summary()

    early = EarlyStopping(monitor='val_binary_accuracy', min_delta=0, patience=20, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.1, patience=9, verbose=1, epsilon=1e-4, mode='min')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=50,
                        verbose=2,
                        callbacks=[early],
                        validation_data=(x_val, y_val),
                        #validation_split=split
                        )

    return model, history
# CNN migration model
def cnn_transfer_model(x_train, y_train, x_val, y_val, frame_size, class_num, batch_size, split):
    base_model = load_model('cnn_CWRU_model_50.h5')
    #base_model = load_model('cnn_gearbox_model_50.h5')
    x = base_model.get_layer('fc_1').output
    x = Dropout(0.5, name='dropout')(x)
    output = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    ''''''
    # 只fine-turning最后一层
    for layer in model.layers[:-3]:
        layer.trainable = False
    for layer in model.layers[-3:]:
        print(layer)
        layer.trainable = True

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    #model.summary()

    early = EarlyStopping(monitor='val_binary_accuracy', min_delta=0, patience=20, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.1, patience=9, verbose=1, epsilon=1e-4, mode='min')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=50,
                        verbose=2,
                        callbacks=[early, ],
                        validation_data=(x_val, y_val),
                        #validation_split=split,
                        )
    return model, history
# MLP Base model
def mlp_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split):
    la = 0.001
    model = Sequential()
    model.add(Dense(512, input_dim=frame_size, activation='relu', kernel_regularizer=l2(la), name='fc'))
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(la), name='dense'))
    model.add(Dense(class_num, activation='softmax', name='output_layer'))
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
#MLP migration model
def mlp_transfer_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split):
    #base_model = load_model('mlp_CWRU_model_50.h5')
    base_model = load_model('mlp_gearbox_model_50.h5')
    x = base_model.get_layer('fc').output
    output = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    '''
    # 只fine-turning最后一层
    for layer in model.layers[:-1]:
        layer.trainable = False
    for layer in model.layers[-1:]:
        layer.trainable = True
    '''
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
    '48k_Drive_End_Bearing_Fault_Data/Ball_Fault/122_1796.csv', '48k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/109_1796.csv',
    '48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/135_1796.csv', '48k_Normal_Baseline_Data/97_1796.csv', 10000
    #'48k_Drive_End_Bearing_Fault_Data/Ball_Fault/192_1725.csv', '48k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/177_1726.csv',
    #'48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/138_1725.csv', '48k_Normal_Baseline_Data/100_1725.csv', 10000
]
work_condition_2_path_list = [
    '12k_Drive_End_Bearing_Fault_Data/Ball_Fault/118_1796.csv', '12k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/169_1796.csv',
    '12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/130_1796.csv', '48k_Normal_Baseline_Data/97_1796.csv', 20000
    #'12k_Drive_End_Bearing_Fault_Data/Ball_Fault/188_1724.csv', '12k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/172_1728.csv',
    #'12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/133_1725.csv', '48k_Normal_Baseline_Data/100_1725.csv', 20000
]
work_condition_3_path_list = [
    '12k_Fan_End_Bearing_Fault_Data/Ball_Fault/290_1796.csv', '12k_Fan_End_Bearing_Fault_Data/Inner_Race_Fault/270_1796.csv',
    '12k_Fan_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/294_1797.csv', '48k_Normal_Baseline_Data/97_1796.csv', 30000
]
work_condition_4_path_list = [
    '48k_Drive_End_Bearing_Fault_Data/Ball_Fault/122_1796.csv', '48k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/109_1796.csv',
    '48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/135_1796.csv', '48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Opposite_Load_Zone/161_1796.csv',
    '48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Orthogonal_To_Load_Zone/250_1796.csv', '48k_Normal_Baseline_Data/97_1796.csv', 40000
    #'48k_Drive_End_Bearing_Fault_Data/Ball_Fault/192_1725.csv', '48k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/177_1726.csv',
    #'48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/138_1725.csv', '48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Opposite_Load_Zone/164_1723.csv',
    #'48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Orthogonal_To_Load_Zone/151_1724.csv', '48k_Normal_Baseline_Data/100_1725.csv', 40000
]
work_condition_5_path_list = [
    '12k_Drive_End_Bearing_Fault_Data/Ball_Fault/118_1796.csv', '12k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/169_1796.csv',
    '12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/130_1796.csv', '12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Opposite_Load_Zone/258_1796.csv',
    '12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Orthogonal_To_Load_Zone/246_1796.csv', '48k_Normal_Baseline_Data/97_1796.csv', 50000
    #'12k_Drive_End_Bearing_Fault_Data/Ball_Fault/188_1724.csv', '12k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/172_1728.csv',
    #'12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/133_1725.csv', '12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Opposite_Load_Zone/160_1724.csv',
    #'12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Orthogonal_To_Load_Zone/147_1725.csv', '48k_Normal_Baseline_Data/100_1725.csv', 50000
]
work_condition_6_path_list = [
    '12k_Fan_End_Bearing_Fault_Data/Ball_Fault/290_1796.csv', '12k_Fan_End_Bearing_Fault_Data/Inner_Race_Fault/270_1796.csv',
    '12k_Fan_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/295_1776.csv', '12k_Fan_End_Bearing_Fault_Data/Outer_Race_Fault_Opposite_Load_Zone/302_1797.csv',
    '12k_Fan_End_Bearing_Fault_Data/Outer_Race_Fault_Orthogonal_To_Load_Zone/310_1796.csv', '48k_Normal_Baseline_Data/97_1796.csv', 60000
]

frame_size = 1000
step = 50
data_size = 400
class_num = 6
batch_size = 512
split = 0.2
#Read data
#data, labels = work_condition_data_4class(frame_size, step, data_size, work_condition_3_path_list)
data, labels = work_condition_data_6class(frame_size, step, data_size, work_condition_6_path_list)
print("work condition 1 total data shape:", data.shape)
print("work condition 1 total labels shape:", labels.shape)


plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 20
plt.subplot(611)
plt.plot(data[0, :], linewidth=1)
plt.ylim((-0.7, 0.7))
plt.title('Ball fault')
plt.subplot(612)
plt.plot(data[500, :], linewidth=1)
plt.ylim((-0.7, 0.7))
plt.title('Inner race fault')
plt.subplot(613)
plt.plot(data[900, :], linewidth=1)
plt.ylim((-0.7, 0.7))
plt.title('Outer race fault centred')
plt.subplot(614)
plt.plot(data[1500, :], linewidth=1)
plt.ylim((-0.7, 0.7))
plt.title('Outer race fault opposite')
plt.subplot(615)
plt.plot(data[1700, :], linewidth=1)
plt.ylim((-0.7, 0.7))
plt.title('Outer race fault orthogonal')
plt.subplot(616)
plt.plot(data[2100, :], linewidth=1)
plt.ylim((-0.7, 0.7))
plt.title('Normal')
plt.show()


# 10% off cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=2)
# training
n = 0
train_acc_list, val_acc_list, test_acc_list = [], [], []
time1 = time.process_time()
for train_index, test_index in kf.split(data):
    time3 = time.process_time()
    print(str(n+1) + "**********************************************************************")
    x_train, y_train = data[train_index], labels[train_index]
    x_test, y_test = data[test_index], labels[test_index]
    print("the number of every class:", np.bincount(np.transpose(np.nonzero(y_train))[:, 1]))
    # upset
    index = [i for i in range(x_train.shape[0])]
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    print("the number of every class:", np.bincount(np.transpose(np.nonzero(y_train))[:, 1]))
    # standardization
    x_train = StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)
    x_val = StandardScaler().fit_transform(x_val)
    # reshape
    x_train = x_train.reshape((-1, frame_size, 1))
    x_test = x_test.reshape((-1, frame_size, 1))
    x_val = x_val.reshape((-1, frame_size, 1))
    # Read CNN model
    model, history = cnn_transfer_model(x_train, y_train, x_val, y_val, frame_size, class_num, batch_size, split)
    #model, history = cnn_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split)
    #Read MLP model
    #model, history = mlp_base_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split)
    #model, history = mlp_transfer_model(x_train, y_train, x_test, y_test, frame_size, class_num, batch_size, split)

    # Output training history information
    print("train acc:", history.history['binary_accuracy'])
    print("val_acc:", history.history['val_binary_accuracy'])
    plt.plot(history.history['binary_accuracy'], label='training acc')
    plt.plot(history.history['val_binary_accuracy'], label='val acc')
    plt.legend()
    plt.show()
    #prediction
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
    print("test acc:", acc)
    print("evaluate acc:", model.evaluate(x_test, y_test, batch_size=x_test.shape[0]))
    print("confuse mat:", confuse_mat)
    n = n + 1

    train_acc_list.append(history.history['binary_accuracy'])
    val_acc_list.append(history.history['val_binary_accuracy'])
    test_acc_list.append(acc)
    time4 = time.process_time()
    print("running time:", str(time4 - time3))
time2 = time.process_time()
#print("mean train acc:", np.mean(np.array(train_acc_list), axis=0))
#print("mean val acc:", np.mean(np.array(val_acc_list), axis=0))
print("train acc list:", train_acc_list)
print("val acc list:", val_acc_list)
print("test acc list:", test_acc_list)
print("mean test acc:", np.mean(test_acc_list))
print("total running time:", str(time2 - time1))