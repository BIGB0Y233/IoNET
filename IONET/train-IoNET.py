from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import numpy as np
import matplotlib.pyplot as plt
import argparse


from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from sklearn.utils import shuffle

from time import time


from dataset import *
from util import *
from model_IoNET import *

#import model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oxiod', 'euroc'], help='Training dataset name (\'oxiod\' or \'euroc\')')
    parser.add_argument('output', help='Model output name')
    args = parser.parse_args(['oxiod','output_data'])
    #args = parser.parse_args()

    np.random.seed(0)

    window_size = 200
    stride = 10

    x_gyro = []
    x_acc = []

    y_delta_p = []
    y_delta_q = []

    imu_data_filenames = []
    gt_data_filenames = []

    if args.dataset == 'oxiod':
        imu_data_filenames.append('imu1.csv')
        imu_data_filenames.append('imu2.csv')
        
        gt_data_filenames.append('vi1.csv')
        gt_data_filenames.append('vi2.csv')
       

    for i, (cur_imu_data_filename, cur_gt_data_filename) in enumerate(zip(imu_data_filenames, gt_data_filenames)):
        if args.dataset == 'oxiod':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)

        [cur_x_gyro, cur_x_acc], [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data, window_size, stride)

        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)

        y_delta_p.append(cur_y_delta_p)
        y_delta_q.append(cur_y_delta_q)

    x_gyro = np.vstack(x_gyro)
    x_acc = np.vstack(x_acc)

    y_delta_p = np.vstack(y_delta_p)
    y_delta_q = np.vstack(y_delta_q)

    x_gyro, x_acc, y_delta_p, y_delta_q = shuffle(x_gyro, x_acc, y_delta_p, y_delta_q)

    #pred_model = create_pred_model_6d_quat(window_size)
    pred_model = step_length_model(window_size)
    train_model = create_train_model_6d_quat(pred_model, window_size)
    train_model.compile(optimizer=Adam(0.0001), loss=None)

    model_checkpoint = ModelCheckpoint('model_checkpoint_IoNET.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    history = train_model.fit([x_gyro, x_acc, y_delta_p, y_delta_q], epochs=100, batch_size=32, verbose=1, callbacks=[model_checkpoint, tensorboard], validation_split=0.1)

    train_model = load_model('model_checkpoint_IoNET.hdf5', custom_objects={'CustomMultiLossLayer':CustomMultiLossLayer}, compile=False)

    #pred_model = create_pred_model_6d_quat(window_size)
    pred_model = step_length_model(window_size)
    pred_model.set_weights(train_model.get_weights()[:-2])
    pred_model.save('%s.hdf5' % args.output)

    plt.figure()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()