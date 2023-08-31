import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

x_train_savepath = '/home/fanxuan/example_data/x_train.npy'
y_train_savepath = '/home/fanxuan/example_data/y_train.npy'


x_test_savepath = '/home/fanxuan/example_data/x_test.npy'
y_test_savepath = '/home/fanxuan/example_data/y_test.npy'


max_dataxy = 2.331058047504005
max_datav = 1.9674335945274706
max_dataa = 59.02300783582412
max_displacement = 8.994006656252493
'''train_path = '/home/fanxuan/Open-Pose-Keras/dataset/cooked/'
train_txt = '/home/fanxuan/Open-Pose-Keras/dataset/folds/split_absent_normal_3_41/train_00.txt'
x_train_savepath = '/home/fanxuan/Open-Pose-Keras/dataset/cooked_subsequences_0500_0100/x_train.npy'
y_train_savepath = '/home/fanxuan/Open-Pose-Keras/dataset/cooked_subsequences_0500_0100/y_train.npy'

test_path =  '/home/fanxuan/Open-Pose-Keras/dataset/cooked/'
test_txt = '/home/fanxuan/Open-Pose-Keras/dataset/folds/split_absent_normal_3_41/test_00.txt'
x_test_savepath = '/home/fanxuan/Open-Pose-Keras/dataset/cooked/x_test.npy'
y_test_savepath = '/home/fanxuan/Open-Pose-Keras/dataset/cooked/y_test.npy'

val_path =  '/home/fanxuan/Open-Pose-Keras/dataset/cooked/'
val_txt = '/home/fanxuan/Open-Pose-Keras/dataset/folds/split_absent_normal_3_41/val_00.txt'
x_val_savepath = '/home/fanxuan/Open-Pose-Keras/dataset/cooked/x_val.npy'
y_val_savepath = '/home/fanxuan/Open-Pose-Keras/dataset/cooked/y_val.npy'

data_path = '/home/fanxuan/Open-Pose-Keras/dataset/cooked_subsequences_1000_0200/'
'''
max_dataxy = 8.972594055715874
max_datav = 20.703786499551647
max_dataa = 158.0717627406676
max_displacement = 4.700494032938104

def find_minmax(data_path):
    file_list = os.listdir(data_path)
    max_dataxy,min_dataxy = [],[]
    max_datav,min_datav = [],[]
    max_dataa,min_dataa = [],[]
    max_data,min_data = [],[]
    for file_name in file_list:
        data_name = os.path.join(data_path,file_name,'raw_skel.npy')
        dataxy = np.load(data_name)[:2,:,:]
        max_dataxy.append(np.max(dataxy))
        min_dataxy.append(np.min(dataxy))
    print(np.max(max_dataxy))
    print(np.min(min_dataxy))
    for file_name in file_list:
        data_name = os.path.join(data_path,file_name,'raw_skel.npy')
        datav = np.load(data_name)[2:4,:,:]
        max_datav.append(np.max(datav))
        min_datav.append(np.min(datav))
    print(np.max(max_datav))
    print(np.min(min_datav))
    for file_name in file_list:
        data_name = os.path.join(data_path,file_name,'raw_skel.npy')
        dataa = np.load(data_name)[4:6,:,:]
        max_dataa.append(np.max(dataa))
        min_dataa.append(np.min(dataa))
    print(np.max(max_dataa))
    print(np.min(min_dataa))
    for file_name in file_list:
        data_name = os.path.join(data_path,file_name,'raw_skel.npy')
        data = np.load(data_name)[6,:,:]
        max_data.append(np.max(data))
        min_data.append(np.min(data))
    print(np.max(max_data))
    print(np.min(min_data))
#find_minmax(data_path)


def generateds(n_sl, n_st, path, txt):
    #n_sl = 100
    #n_st = 500
    f = open(txt, 'r')  
    contents = f.readlines()[1:]
    f.close() 
    x, y_ ,labels= [], [] ,[]
    for content in contents:  
        value = content.split(',') 
        img_path = Path(path, value[0],'raw_skel.npy')
        img = np.load(img_path)[:7,:,:]  
        #img = np.array(img.convert('L')) 
        img[:2,:,:] = img[:2,:,:] / max_dataxy
        img[2:4,:,:] = img[2:4,:,:] / max_datav
        img[4:6,:,:] = img[4:6,:,:] / max_dataa
        img[6,:,:] = img[6,:,:] / max_displacement
        j = 0
        while j+n_st <= img.shape[1]:
            x.append(img[:,j:j+n_st,:])
            y_.append(value[1]) 
            labels.append(value[0]) 
            j = j+n_sl
        #x.append(img)  
        #y_.append(value[1]) 
        #print('loading : ' + content)  

    x = np.array(x)  
    y_ = np.array(y_)
    labels = np.array(labels) 
    y_ = y_.astype(np.int64)  
    print("x shape", x.shape)
    print("y_ shape", y_.shape)
    return x, y_ ,labels


def load_data(an_id, n_sl, n_st):
    data_root = '/home/fanxuan/Open-Pose-Keras/dataset_1st/'
    pose_data_root = Path(data_root, f'cooked_ori')
    fold_dir = Path(data_root, 'folds/split_absent_normal_ori_3_41/')
    #pose_data_root = Path(data_root, f'cooked_subsequences_1000_0200')
    #fold_dir = Path(data_root, f'folds/split_2class_91')
    train_label_file = Path(fold_dir, f'train_{an_id:02}.txt')
    val_label_file = Path(fold_dir, f'val_{an_id:02}.txt')
    test_label_file = Path(fold_dir, f'test_{an_id:02}.txt')
    if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
            x_test_savepath) and os.path.exists(y_test_savepath):
        print('-------------Load Datasets-----------------')
        '''x_train_save = np.load(x_train_savepath)
        y_train = np.load(y_train_savepath)
        x_test_save = np.load(x_test_savepath)
        y_test = np.load(y_test_savepath)
        x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
        x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))'''

    else:
        print('-------------Generate Datasets-----------------')
        print("train")
        x_train, y_train, labels_train = generateds(n_sl, n_st,pose_data_root, train_label_file)
        print("test")
        x_test, y_test, labels_test = generateds(n_sl, n_st,pose_data_root, test_label_file)
        print("val")
        x_val, y_val, labels_val = generateds(n_sl, n_st,pose_data_root, val_label_file)

        '''print('-------------Save Datasets-----------------')
        x_train_save = np.reshape(x_train, (len(x_train), -1))
        x_test_save = np.reshape(x_test, (len(x_test), -1))
        np.save(x_train_savepath, x_train_save)
        np.save(y_train_savepath, y_train)
        np.save(x_test_savepath, x_test_save)
        np.save(y_test_savepath, y_test)'''
    return x_train, y_train,labels_train, x_test, y_test, labels_test, x_val, y_val, labels_val

#find_minmax(data_path)

