import torch as th
import torch.nn as nn
from sklearn import metrics
import numpy as np
from torch.utils.data import DataLoader
from utils.utils import vstack, ConfigLoader, get_root_dir, load_model, use_devices, get_infant_id
from pathlib import Path
from data.dataset import ClipDataset
import argparse



def cross_entropy_error(y,y_true):
    delta=1e-7  
    return -np.sum(y_true*np.log(y+delta))
    
    
def evaluate(net, dataset, suffix_len=6, device="cuda"):
    net.eval()

    scores = None
    dir_name_list = []
    infant_list = []
    y_true = []
    data_loader = DataLoader(dataset=dataset, batch_size=64, num_workers=8, shuffle=False)

    # Compute scores on sub-sequences
    for i, batch_data in enumerate(data_loader):
        batch_X_cpu, batch_y_true, batch_dir_names = batch_data
        batch_X_gpu = batch_X_cpu.to(device, dtype=th.float)
        batch_scores = net(x=batch_X_gpu)
        batch_scores = th.softmax(batch_scores, dim=1)
        batch_scores = batch_scores.detach().cpu().numpy()

        dir_name_list.extend(list(batch_dir_names))
        y_true.extend(list(batch_y_true))

        scores = vstack(scores, batch_scores)

    infant_score_dict = dict()
    infant_label_dict = dict()
    # Summary scores of sub-sequences
    for i, dir_name in enumerate(dir_name_list):
        # dir_name = xxxxx_xxx_sub_y, where xxxx_xxx = infant_id, y = sub sequence id
        # remove _sub_y from the dir_name to get the infant_id
        infant_id = get_infant_id(dir_name, suffix_len)
        sub_sequence_score = scores[i, 1]
        if infant_id not in infant_score_dict.keys():
            infant_score_dict[infant_id] = [sub_sequence_score]
            infant_label_dict[infant_id] = y_true[i]
        else:
            infant_score_dict[infant_id].append(sub_sequence_score)
            if infant_label_dict[infant_id] != y_true[i]:
                raise Exception("Voting metrics: labels not consistent on sub-sequences!!!")
    agg_scores = []
    agg_y_true = []
    for infant_id in infant_score_dict.keys():
        infant_list.append(infant_id)
        # Score of the whole sequence is the max scores of sub-sequences
        agg_scores.append(max(infant_score_dict[infant_id]))
        agg_y_true.append(infant_label_dict[infant_id])

    # Accuracy on whole skeleton level (aggregation of sub-sequences)
    agg_scores = np.asarray(agg_scores)
    agg_y_true = np.asarray(agg_y_true)
    agg_roc_auc = metrics.roc_auc_score(agg_y_true, agg_scores, average="weighted")
    return agg_roc_auc
def evaluate_CM(net, dataset, suffix_len=6, device="cuda"):
    net.eval()

    scores = None
    dir_name_list = []
    infant_list = []
    y_true = []
    data_loader = DataLoader(dataset=dataset, batch_size=64, num_workers=8, shuffle=False)

    # Compute scores on sub-sequences
    for i, batch_data in enumerate(data_loader):
        batch_X_cpu, batch_y_true, batch_dir_names = batch_data
        batch_X_gpu = batch_X_cpu.to(device, dtype=th.float)
        #print(batch_X_gpu.shape)
        batch_scores = net(x=batch_X_gpu)
        #print(batch_scores.shape)
        batch_scores = th.softmax(batch_scores, dim=1)
        #print(batch_scores.shape)
        batch_scores = batch_scores.detach().cpu().numpy()
        #print(batch_dir_names)
        dir_name_list.extend(list(batch_dir_names))
        y_true.extend(list(batch_y_true))

        scores = vstack(scores, batch_scores)
        #print(scores.shape)
    infant_score_dict = dict()
    infant_label_dict = dict()
    # Summary scores of sub-sequences
    for i, dir_name in enumerate(dir_name_list):
        # dir_name = xxxxx_xxx_sub_y, where xxxx_xxx = infant_id, y = sub sequence id
        # remove _sub_y from the dir_name to get the infant_id
        infant_id = get_infant_id(dir_name, suffix_len)
        sub_sequence_score = scores[i, 1]
        a = scores[i, :]
        if infant_id not in infant_score_dict.keys():
            infant_score_dict[infant_id] = [sub_sequence_score]
            infant_label_dict[infant_id] = y_true[i]
        else:
            #print('if else')
            #(sub_sequence_score)
            infant_score_dict[infant_id].append(sub_sequence_score)
            if infant_label_dict[infant_id] != y_true[i]:
                raise Exception("Voting metrics: labels not consistent on sub-sequences!!!")
    agg_scores = []
    agg_y_true = []
    for infant_id in infant_score_dict.keys():
        infant_list.append(infant_id)
        # Score of the whole sequence is the max scores of sub-sequences
        agg_scores.append(max(infant_score_dict[infant_id]))
        agg_y_true.append(infant_label_dict[infant_id])

    # Accuracy on whole skeleton level (aggregation of sub-sequences)


    agg_scores = np.asarray(agg_scores)
    agg_y_true = np.asarray(agg_y_true)
    avg_loss = cross_entropy_error(agg_scores,agg_y_true)
    #print(avg_loss)
    agg_scores[agg_scores<0.5]=0
    agg_scores[agg_scores>=0.5]=1
    agg_roc_auc = metrics.confusion_matrix(agg_y_true, agg_scores)
    c0 = agg_roc_auc[0]
    c1 = agg_roc_auc[1]
    avg_acc = (c0[0]/(c0[0]+c0[1])+c1[1]/(c1[0]+c1[1]))/2
    #print(avg_acc)
    return agg_roc_auc, avg_acc, avg_loss 


def evaluate_CM_validation(net, dataset, suffix_len=6, device="cuda"):
    net.eval()

    scores = None
    dir_name_list = []
    infant_list = []
    y_true = []
    data_loader = DataLoader(dataset=dataset, batch_size=64, num_workers=8, shuffle=False)

    # Compute scores on sub-sequences
    for i, batch_data in enumerate(data_loader):
        batch_X_cpu, batch_y_true, batch_dir_names = batch_data
        batch_X_gpu = batch_X_cpu.to(device, dtype=th.float)
        #print(batch_X_gpu.shape)
        batch_scores = net(x=batch_X_gpu)
        #print(batch_scores.shape)
        batch_scores = th.softmax(batch_scores, dim=1)
        #print(batch_scores.shape)
        batch_scores = batch_scores.detach().cpu().numpy()
        #print(batch_dir_names)
        dir_name_list.extend(list(batch_dir_names))
        y_true.extend(list(batch_y_true))

        scores = vstack(scores, batch_scores)
        #print(scores.shape)
    infant_score_dict = dict()
    infant_label_dict = dict()
    # Summary scores of sub-sequences
    for i, dir_name in enumerate(dir_name_list):
        # dir_name = xxxxx_xxx_sub_y, where xxxx_xxx = infant_id, y = sub sequence id
        # remove _sub_y from the dir_name to get the infant_id
        infant_id = get_infant_id(dir_name, suffix_len)
        sub_sequence_score = scores[i, 1]
        a = scores[i, :]
        if infant_id not in infant_score_dict.keys():
            infant_score_dict[infant_id] = [sub_sequence_score]
            infant_label_dict[infant_id] = y_true[i]
        else:
            #print('if else')
            #(sub_sequence_score)
            infant_score_dict[infant_id].append(sub_sequence_score)
            if infant_label_dict[infant_id] != y_true[i]:
                raise Exception("Voting metrics: labels not consistent on sub-sequences!!!")
    agg_scores = []
    agg_y_true = []
    for infant_id in infant_score_dict.keys():
        infant_list.append(infant_id)
        # Score of the whole sequence is the max scores of sub-sequences
        agg_scores.append(max(infant_score_dict[infant_id]))
        agg_y_true.append(infant_label_dict[infant_id])

    # Accuracy on whole skeleton level (aggregation of sub-sequences)


    agg_scores = np.asarray(agg_scores)
    agg_y_true = np.asarray(agg_y_true)
    infant_list = np.asarray(infant_list)
    avg_loss = cross_entropy_error(agg_scores,agg_y_true)
    #print(avg_loss)
    agg_scores[agg_scores<0.5]=0
    agg_scores[agg_scores>=0.5]=1
    print('TF')
    TF = infant_list[np.where((agg_y_true==0)&(agg_scores==1))]
    print(infant_list[np.where((agg_y_true==0)&(agg_scores==1))])
    print('FF')
    print(infant_list[np.where((agg_y_true==1)&(agg_scores==0))])
    FF = infant_list[np.where((agg_y_true==1)&(agg_scores==0))]
    agg_roc_auc = metrics.confusion_matrix(agg_y_true, agg_scores)
    c0 = agg_roc_auc[0]
    c1 = agg_roc_auc[1]
    avg_acc = (c0[0]/(c0[0]+c0[1])+c1[1]/(c1[0]+c1[1]))/2
    #print(avg_acc)
    return agg_roc_auc, avg_acc, avg_loss,TF,FF
    
        
def evaluate_3C(net, dataset, suffix_len=6, device="cuda"):
    net.eval()

    scores = None
    dir_name_list = []
    infant_list = []
    y_true = []
    data_loader = DataLoader(dataset=dataset, batch_size=64, num_workers=8, shuffle=False)

    # Compute scores on sub-sequences
    for i, batch_data in enumerate(data_loader):
        batch_X_cpu, batch_y_true, batch_dir_names = batch_data
        batch_X_gpu = batch_X_cpu.to(device, dtype=th.float)

        batch_scores = net(x=batch_X_gpu)

        batch_scores = th.softmax(batch_scores, dim=1)

        batch_scores = batch_scores.detach().cpu().numpy()

        dir_name_list.extend(list(batch_dir_names))
        y_true.extend(list(batch_y_true))

        scores = vstack(scores, batch_scores)
    infant_score_dict = dict()
    infant_label_dict = dict()
    # Summary scores of sub-sequences
    for i, dir_name in enumerate(dir_name_list):
        # dir_name = xxxxx_xxx_sub_y, where xxxx_xxx = infant_id, y = sub sequence id
        # remove _sub_y from the dir_name to get the infant_id
        infant_id = get_infant_id(dir_name, suffix_len)
        sub_sequence_score = scores[i, :]
        if infant_id not in infant_score_dict.keys():
            infant_score_dict[infant_id] = [sub_sequence_score]
            infant_label_dict[infant_id] = y_true[i]
        else:
            print('if else')
            print(sub_sequence_score)
            infant_score_dict[infant_id].append(sub_sequence_score)
            print(infant_score_dict[infant_id])
            if infant_label_dict[infant_id] != y_true[i]:
                raise Exception("Voting metrics: labels not consistent on sub-sequences!!!")
    agg_scores = []
    agg_y_true = []
    for infant_id in infant_score_dict.keys():
        infant_list.append(infant_id)
        # Score of the whole sequence is the max scores of sub-sequences
        #print(infant_score_dict[infant_id])
        max_s = max(infant_score_dict[infant_id])
        agg_scores.append(np.where(max_s==max(max_s))[0])
        agg_y_true.append(infant_label_dict[infant_id])

    # Accuracy on whole skeleton level (aggregation of sub-sequences)
    agg_scores = np.asarray(agg_scores)
    agg_y_true = np.asarray(agg_y_true)
    #print(agg_scores)
    agg_roc_auc = metrics.confusion_matrix(agg_y_true, agg_scores)
    return agg_roc_auc

def main():
    parser = argparse.ArgumentParser(description="Parse the parameters for training the model.")
    parser.add_argument('--fold-id', type=int, default=0, help='fold id')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--clip-size', type=int, default=30, help='Size of each block.')
    parser.add_argument('--stride', type=int, default=20, help='Size of each block.')
    parser.add_argument('--length', type=int, default=1000, help='Length of the skeleton to use')
    parser.add_argument('--epoch', type=int, default=10, help='Epoch of the saved model.')
    parser.add_argument('--suffix_len', type=int, default=6, help='Suffix length of the infant dir.')

    args = parser.parse_args()
    args.use_cuda = not args.no_cuda and th.cuda.is_available()
    device = th.device("cuda" if args.use_cuda else "cpu")

    conf = ConfigLoader().config
    data_root = conf['data_root']
    pose_data_root = Path(data_root, f'sub_sequences_1000_0200')
    fold_dir = Path(data_root, f'folds/random_split')
    val_label_file = Path(fold_dir, f'test_{args.fold_id:02}.txt')

    val_set = ClipDataset(data_root=pose_data_root, clip_size=args.clip_size, stride=args.stride,
                          length=args.length, label_file_path=val_label_file, suffix_len=args.suffix_len)

    root_dir = get_root_dir()
    model_path = Path(root_dir, f'checkpoints/checkpoint_epoch_{args.epoch:03}.pth')
    net = load_model(model_path)
    net = use_devices(net, device, multi_gpus=False)

    roc_auc = evaluate(net, val_set, suffix_len=args.suffix_len, device=device)
    print("roc_auc:", roc_auc)
    return 0

if __name__ == '__main__':
    main()