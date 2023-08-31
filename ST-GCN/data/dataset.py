from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from utils.utils import get_infant_id
import pandas as pd 
import random
from random import randint


def load_pose_sequence(file_path, length=None):
    try:
        data = np.load(file_path, allow_pickle=True)
        if length is None:
            length = data.shape[1]
            data = data[:, :length, :]
            #print(data.shape)
        else:
            data_length = data.shape[1]
            step = (data_length-length)//50
            index_num=randint(0,49)
            start = index_num*step
            end = start + length
            if end>data_length:
                end = data_length
                start = end - length
            data = data[:, start:end, :] 
            #print(data.shape)      
    except Exception as e:
        raise Exception(f"Error in loading pose sequence: {e}!!!")
    return data

def load_labels_old(label_file_path):
    label_dict = dict()
    f = open(label_file_path, 'r')
    lines = f.readlines()[1:]
    f.close()

    for line in lines:
        infant_id, label = line.split(',')
        label_dict[infant_id] = int(label)
    return label_dict

def load_labels(label_file_path):
    label_dict = dict()
    df = pd.read_csv(label_file_path)
    infant_ids = df["ID - GMA"]
    infant_labels = df["Label"]
    for ind in range(len(infant_ids)):
        infant_id, label = infant_ids[ind],infant_labels[ind]
        label_dict[infant_id] = int(label)
    return label_dict
    

def split_to_clips(pose_seq, clip_size, stride):
    clip_list = []
    length = pose_seq.shape[1]
    for start in range(0, length, stride):
        end = start + clip_size
        clip = np.zeros(pose_seq[:, start:end, :].shape)
        clip[:, :, :] = pose_seq[:, start:end, :]
        if clip.shape[1] < clip_size:
            continue
        clip_list.append(clip)
    clip_list = np.stack(clip_list, axis=0)
    return clip_list


class ClipDataset(Dataset):
    def __init__(self, data_root, label_file_path, clip_size, stride, length,
                 channels=7, file_name='pose_sequence.npy', suffix_len=0):
        super(ClipDataset, self).__init__()
        print("data_root:", data_root)
        self.data_paths = []
        self.label_list = []
        self.infant_list = []
        label_dict = load_labels_old(label_file_path)
        infant_dir_list = [e for e in Path(data_root).rglob('*') if e.is_dir()]
        infant_dir_list = sorted(infant_dir_list)
        infant_dir_list.extend(infant_dir_list)
        #print(infant_dir_list) 
        for infant_dir in infant_dir_list:
            # dir_name = xxxxx_xxx_sub_y, where xxxx_xxx = infant_id, y = sub sequence id
            dir_name = str(infant_dir.stem)
            # remove _sub_y from the dir_name to get the infant_id
            # infant_id = dir_name[:-6]
            infant_id = get_infant_id(dir_name, suffix_len=suffix_len)
            #print(infant_id[:-7])
            if infant_id[:-7] in label_dict.keys():
                data_path = Path(data_root, dir_name, f'{file_name}')
                if not data_path.exists():
                    continue
                self.data_paths.append(data_path)
                label = label_dict[infant_id[:-7]]
                self.label_list.append(label)
                self.infant_list.append(dir_name)
        self.clip_size = clip_size
        self.stride = stride
        self.length = length
        self.channels = channels

    def __getitem__(self, index):
        data_file_path = self.data_paths[index]
        label = self.label_list[index]
        infant_id = self.infant_list[index]
        pose_sequence = load_pose_sequence(file_path=data_file_path)[:self.channels, :, :]
        clip_list = split_to_clips(pose_sequence, clip_size=self.clip_size, stride=self.stride)
        return clip_list, label, infant_id

    def __len__(self):
        return len(self.data_paths)
        
