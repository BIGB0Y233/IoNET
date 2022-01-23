import numpy as np
import pandas as pd
import quaternion  #四元素，a+bi+cj+dk
import scipy.interpolate#插值
from tensorflow.keras.utils import Sequence#model.fit()传入训练数据数据，fit()接收多种类型的数据




def load_oxiod_dataset(imu_data_filename, gt_data_filename):
    imu_data = pd.read_csv(imu_data_filename).values#导入imu数据
    gt_data = pd.read_csv(gt_data_filename).values＃导入gt数据

    imu_data = imu_data[1200:-300]#读取imu从正数据的第1200个数据读取到倒数300个的数据
    gt_data = gt_data[1200:-300]＃读取gt从正数据的第1200个数据读取到倒数300个的数据

    gyro_data = imu_data[:, 4:7]#重力g的数据，取二维数组中第m到n-1列数据，m为4，n为7下面行相同
    acc_data = imu_data[:, 10:13]#加速度a的数据，取二维数组中的第10到12列的数据
    
    pos_data = gt_data[:, 2:5]＃位置position的数据，取二维数组中的第2列到第4列的数据
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)#将第8列的数据和第5到7列的数据进行行拼接axis=1为行为0为列拼接

    print("load_oxiod")
    print(pos_data.shape,gyro_data.shape[0])
    return gyro_data, acc_data, pos_data, ori_data#返回重力，加速度，位置，原始数据的数据


def load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size=200, stride=10):
    #gyro_acc_data = np.concatenate([gyro_data, acc_data], axis=1)

    init_p = pos_data[window_size//2 - stride//2, :]#py中／／表示向下取整，窗大小，和步长大小
    init_q = ori_data[window_size//2 - stride//2, :]＃初始数据

    #x = []
    x_gyro = []＃重力
    x_acc = []＃加速度
    y_delta_p = []＃变化的位置
    y_delta_q = []＃变化的

    print(gyro_data.shape[0],pos_data.shape)
    for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):#遍历整个数据，shape[0]表示垂直大小
        #x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])
        x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])#获取x轴g，数据长度为窗大小
        x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])#获取x轴加速度，数据长度为窗大小
        
        p_a = pos_data[idx + window_size//2 - stride//2, :]#位置a
        p_b = pos_data[idx + window_size//2 + stride//2, :]#位置b

        q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])#四元数法位置a
        q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

        delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T#矩阵相乘，变化的位置p

        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)#共轭后转化为四元数

        y_delta_p.append(delta_p)
        y_delta_q.append(quaternion.as_float_array(delta_q))


    #x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))#转化格式
    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
    y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
    y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))

    #return x, [y_delta_p, y_delta_q], init_p, init_q
    return [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q#返还x轴和y轴的g和加速度

