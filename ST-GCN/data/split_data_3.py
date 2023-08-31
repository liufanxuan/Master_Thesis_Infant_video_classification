import sys
sys.path.insert(0,'/home/fanxuan/stam-main/src/utils/')
sys.path.insert(0,'/home/fanxuan/stam-main/src/')
from pathlib import Path
import numpy as np
import argparse
from data.dataset import load_labels
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from utils.utils import ConfigLoader

def stratified_split_random(target_dir, X, y, cv=5, random_state=42):
    df = pd.DataFrame()
    df['infant_id'] = X
    df['label'] = y
    df = df.sort_values(by='infant_id', ascending=1)
    X_data = np.asarray(df['infant_id'])
    print(X_data)
    y_label = np.asarray(df['label'])
    print(y_label)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=random_state)
    print(sss)
    fold_idx = 0
    for train_index, test_val_index in sss.split(X_data, y_label):
        print(train_index)
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_label[train_index], y_label[test_index]

        train_file_path = Path(target_dir, f'train_{fold_idx:02}.txt')
        f_train = open(str(train_file_path), 'w')
        f_train.write('infant_id,label\n')
        for i, infant_id in enumerate(X_train):
            label = y_train[i]
            f_train.write(f'{infant_id},{label}\n')
        f_train.close()

        test_file_path = Path(target_dir, f'test_{fold_idx:02}.txt')
        f_test = open(test_file_path, 'w')
        f_test.write('infant_id,label\n')
        for i, infant_id in enumerate(X_test):
            label = y_test[i]
            f_test.write(f'{infant_id},{label}\n')
        f_test.close()
        fold_idx += 1
        
def load_data_split(data_root, train_label_dict, test_label_dict, val_label_dict):
    data_paths = Path(data_root).rglob('raw_skel.npy')

    train_infant_list = []
    train_labels = []
    test_infant_list = []
    test_labels = []
    val_infant_list = []
    val_labels = []
    
    for data_path in data_paths:

        if not data_path.exists():
            continue
        data_dir = data_path.resolve().parents[0]
        dir_name = str(data_dir.stem)
        infant_id = dir_name[:13].split(' ')[1].split('_')[0]
        if int(infant_id) in train_label_dict.keys():
            train_infant_list.append(dir_name)
            train_labels.append(train_label_dict[int(infant_id)])
        elif int(infant_id) in test_label_dict.keys():
            test_infant_list.append(dir_name)
            test_labels.append(test_label_dict[int(infant_id)])
        elif int(infant_id) in val_label_dict.keys():
            val_infant_list.append(dir_name)
            val_labels.append(val_label_dict[int(infant_id)])
        else:
            print('False to load')
            print(dir_name)

    return train_infant_list, train_labels, test_infant_list, test_labels, val_infant_list, val_labels
    
def stratified_split_random_new(target_dir, label_file_path, data_root, cv=5, random_state=42):
    label_dict = dict()
    df = pd.read_csv(label_file_path)
    df = df.sort_values(by="ID - GMA", ascending=1)
    df_shuffled=df.sample(frac=1).reset_index(drop=True)
    infant_ids = np.array(df["ID - GMA"])
    infant_labels = np.array(df["Label"])  
    data_paths = Path(data_root).rglob('praw_skel.npy')

    sss = StratifiedShuffleSplit(n_splits=cv, test_size=0.2, random_state=random_state)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    fold_idx = 0
    for train_index, test_val_index in sss.split(infant_ids, infant_labels):
        id_train, id_test_val = infant_ids[train_index], infant_ids[test_val_index]
        label_train, label_test_val = infant_labels[train_index], infant_labels[test_val_index]   
        for test_index, val_index in sss2.split(id_test_val, label_test_val):
            id_test, id_val = id_test_val[test_index], id_test_val[val_index]
            label_test, label_val = label_test_val[test_index], label_test_val[val_index]             
        print(id_test)
        print(id_val)
        print(id_train)        
        train_label_dict = dict() 
        test_label_dict = dict() 
        val_label_dict = dict() 
        for ind in range(len(id_train)):
            inf, lab = id_train[ind], label_train[ind]
            train_label_dict[inf] = int(lab)
        for ind in range(len(id_test)):
            inf, lab = id_test[ind], label_test[ind]
            test_label_dict[inf] = int(lab)   
        for ind in range(len(id_val)):
            inf, lab = id_val[ind], label_val[ind]
            val_label_dict[inf] = int(lab)     

        train_infant_list, train_labels, test_infant_list, test_labels, val_infant_list, val_labels = load_data_split(data_root, train_label_dict, test_label_dict, val_label_dict)
        train_file_path = Path(target_dir, f'train_{fold_idx:02}.txt')
        test_file_path = Path(target_dir, f'test_{fold_idx:02}.txt')
        val_file_path = Path(target_dir, f'val_{fold_idx:02}.txt')
        creat_rand_dataset(train_file_path, train_infant_list, train_labels)
        creat_rand_dataset(test_file_path, test_infant_list, test_labels)
        creat_rand_dataset(val_file_path, val_infant_list, val_labels)
        fold_idx += 1

def creat_rand_dataset(target_dir, X, y):
    df = pd.DataFrame()
    df['infant_id'] = X
    df['label'] = y
    df = df.sort_values(by='infant_id', ascending=1)
    df_shuffled=df.sample(frac=1).reset_index(drop=True)
    df_shuffled.to_csv(target_dir,index=False,sep=',',encoding='utf_8_sig')
    
def load_data(data_root, label_dict):
    data_paths = Path(data_root).rglob('raw_skel.npy')

    infant_list = []
    labels = []

    for data_path in data_paths:

        if not data_path.exists():
            continue
        data_dir = data_path.resolve().parents[0]
        dir_name = str(data_dir.stem)
        infant_id = dir_name[:13].split(' ')[1].split('_')[0]
        infant_list.append(dir_name)
        if int(infant_id) in label_dict.keys():
            labels.append(label_dict[int(infant_id)])
        else:
            print(dir_name)
            labels.append(0)
    return infant_list, labels

def main():
    parser = argparse.ArgumentParser(description="Parse the parameters for the datasets split.")
    parser.add_argument('--cv', type=int, default=5, help='Number of splits.')
    parser.add_argument('--random-state', type=int, default=0, help='Random state.')
    args = parser.parse_args()
    cv = args.cv
    random_state = args.random_state

    conf = ConfigLoader().config
    data_root = Path(conf['data_root'], f'cooked_ori')
    label_file_path = Path(conf['data_root'], 'Label_absent_normal_ori.csv')
    split_dir = Path(conf['data_root'], f'folds/split_absent_normal_all')
    split_dir.mkdir(parents=True, exist_ok=True)

    stratified_split_random_new(split_dir, label_file_path, data_root)

if __name__ == '__main__':
    main()
