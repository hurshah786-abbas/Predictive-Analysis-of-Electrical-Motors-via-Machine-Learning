import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# CWRU 4 classification conditions
def CWRU_work_condition_data_4class(frame_size, step, data_size, path_list):
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
#CWRU 6 classification conditions
def CWRU_work_condition_data_6class(frame_size, step, data_size, path_list):
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

# Gear 6 classification
def gear_work_condition_data_6class(frame_size, step, data_size, path_list):
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
# gear 8 Classification
def gear_work_condition_data_8class(frame_size, step, data_size, path_list):
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


CWRU_work_condition_1_path_list = [
    '48k_Drive_End_Bearing_Fault_Data/Ball_Fault/122_1796.csv', '48k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/109_1796.csv',
    '48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/135_1796.csv', '48k_Normal_Baseline_Data/97_1796.csv', 10000
]
CWRU_work_condition_2_path_list = [
    '12k_Drive_End_Bearing_Fault_Data/Ball_Fault/118_1796.csv', '12k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/169_1796.csv',
    '12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/130_1796.csv', '48k_Normal_Baseline_Data/97_1796.csv', 20000
]
CWRU_work_condition_3_path_list = [
    '12k_Fan_End_Bearing_Fault_Data/Ball_Fault/290_1796.csv', '12k_Fan_End_Bearing_Fault_Data/Inner_Race_Fault/270_1796.csv',
    '12k_Fan_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/294_1797.csv', '48k_Normal_Baseline_Data/97_1796.csv', 30000
]
CWRU_work_condition_4_path_list = [
    '48k_Drive_End_Bearing_Fault_Data/Ball_Fault/122_1796.csv', '48k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/109_1796.csv',
    '48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/135_1796.csv', '48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Opposite_Load_Zone/161_1796.csv',
    '48k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Orthogonal_To_Load_Zone/250_1796.csv', '48k_Normal_Baseline_Data/97_1796.csv', 40000
]
CWRU_work_condition_5_path_list = [
    '12k_Drive_End_Bearing_Fault_Data/Ball_Fault/118_1796.csv', '12k_Drive_End_Bearing_Fault_Data/Inner_Race_Fault/169_1796.csv',
    '12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/130_1796.csv', '12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Opposite_Load_Zone/258_1796.csv',
    '12k_Drive_End_Bearing_Fault_Data/Outer_Race_Fault_Orthogonal_To_Load_Zone/246_1796.csv', '48k_Normal_Baseline_Data/97_1796.csv', 50000
]
CWRU_work_condition_6_path_list = [
    '12k_Fan_End_Bearing_Fault_Data/Ball_Fault/290_1796.csv', '12k_Fan_End_Bearing_Fault_Data/Inner_Race_Fault/270_1796.csv',
    '12k_Fan_End_Bearing_Fault_Data/Outer_Race_Fault_Centred_On_Load_Zone/294_1797.csv', '12k_Fan_End_Bearing_Fault_Data/Outer_Race_Fault_Opposite_Load_Zone/302_1797.csv',
    '12k_Fan_End_Bearing_Fault_Data/Outer_Race_Fault_Orthogonal_To_Load_Zone/310_1796.csv', '48k_Normal_Baseline_Data/97_1796.csv', 60000
]

gear_work_condition_1_path_list = [
    'helical 1/helical 1_30hz_High_1.txt', 'helical 2/helical 2_30hz_High_1.txt', 'helical 3/helical 3_30hz_High_1.txt', 'helical 4/helical 4_30hz_High_1.txt',
    'helical 5/helical 5_30hz_High_1.txt', 'helical 6/helical 6_30hz_High_1.txt',
]
gear_work_condition_2_path_list = [
    'helical 1/helical 1_35hz_Low_1.txt', 'helical 2/helical 2_35hz_Low_1.txt', 'helical 3/helical 3_35hz_Low_1.txt', 'helical 4/helical 4_35hz_Low_1.txt',
    'helical 5/helical 5_35hz_Low_1.txt', 'helical 6/helical 6_35hz_Low_1.txt',
]
gear_work_condition_3_path_list = [
    'spur 1/spur 1_30hz_High_1.txt', 'spur 2/spur 2_30hz_High_1.txt', 'spur 3/spur 3_30hz_High_1.txt', 'spur 4/spur 4_30hz_High_1.txt', 'spur 5/spur 5_30hz_High_1.txt',
    'spur 6/spur 6_30hz_High_1.txt', 'spur 7/spur 7_30hz_High_1.txt', 'spur 8/spur 8_30hz_High_1.txt',
]
gear_work_condition_4_path_list = [
    'spur 1/spur 1_30hz_Low_1.txt', 'spur 2/spur 2_30hz_Low_1.txt', 'spur 3/spur 3_30hz_Low_1.txt', 'spur 4/spur 4_30hz_Low_1.txt', 'spur 5/spur 5_30hz_Low_1.txt',
    'spur 6/spur 6_30hz_Low_1.txt', 'spur 7/spur 7_30hz_Low_1.txt', 'spur 8/spur 8_30hz_Low_1.txt',
]
gear_work_condition_5_path_list = [
    'spur 1/spur 1_50hz_High_1.txt', 'spur 2/spur 2_50hz_High_1.txt', 'spur 3/spur 3_50hz_High_1.txt', 'spur 4/spur 4_50hz_High_1.txt', 'spur 5/spur 5_50hz_High_1.txt',
    'spur 6/spur 6_50hz_High_1.txt', 'spur 7/spur 7_50hz_High_1.txt', 'spur 8/spur 8_50hz_High_1.txt',
]
gear_work_condition_6_path_list = [
    'spur 1/spur 1_50hz_Low_1.txt', 'spur 2/spur 2_50hz_Low_1.txt', 'spur 3/spur 3_50hz_Low_1.txt', 'spur 4/spur 4_50hz_Low_1.txt', 'spur 5/spur 5_50hz_Low_1.txt',
    'spur 6/spur 6_50hz_Low_1.txt', 'spur 7/spur 7_50hz_Low_1.txt', 'spur 8/spur 8_50hz_Low_1.txt',
]

frame_size = 1000
step = 50
data_size = 400
class_num = 6
# Read CWRU data
#data, labels = CWRU_work_condition_data_4class(frame_size, step, data_size, CWRU_work_condition_3_path_list)
# Read gear data
data, labels = gear_work_condition_data_6class(frame_size, step, data_size, gear_work_condition_2_path_list)

# standardization
data = StandardScaler().fit_transform(data)
# reshape
data = data.reshape((-1, frame_size, 1))


model = load_model('cnn_CWRU_model_50.h5')
model = load_model('mlp_CWRU_model_50.h5')
model = load_model('cnn_gearbox_model_50.h5')
model = load_model('mlp_gearbox_model_50.h5')

#prediction
predict = model.predict(data, batch_size=1, verbose=1)
correct_num = 0
confuse_mat = np.array(np.zeros((class_num, class_num)))    # 混淆矩阵
for i in range(predict.shape[0]):
    indexpr = np.where(predict[i, :] == np.max(predict[i, :]))
    indextest = np.where(labels[i, :] == np.max(labels[i, :]))
    if indexpr == indextest:
        correct_num = correct_num + 1
    confuse_mat[indexpr[0][0]-1, indextest[0][0]-1] += 1
acc = correct_num/predict.shape[0]
print("test acc:", acc)
print("confuse mat:", confuse_mat)






