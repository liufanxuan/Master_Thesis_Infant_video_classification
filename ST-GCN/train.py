import torch.nn as nn
import torch as th
import argparse
from model.stam import STAM
from utils.utils import ConfigLoader, use_devices, save_model, get_root_dir, init_seed
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
from validate import evaluate, evaluate_CM, evaluate_3C, evaluate_CM_validation
from utils.imbalanced import ImbalancedSampler
from data.dataset import ClipDataset
import mlflow as mf
import pandas as pd

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(args, net, train_set, val_set, save_dir, an_id):
    weights = [1.0,7.0]

    device = th.device("cuda" if args.use_cuda else "cpu")
    class_weights = th.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = nn.CrossEntropyLoss()
    net = use_devices(net, device)

    # if args.optimizer == "Adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    max_val_roc_auc = -1

    mf.set_experiment(experiment_name="stam_experiment")
    with mf.start_run():
        mf.log_param('fold_id', args.fold_id)
        mf.log_param("parameters", vars(args))
        epoch_df = pd.DataFrame()
        val_roc_auces, val_acces, train_roc_auces, train_acces, epoches, TFs, FFs = [],[],[],[],[],[],[]
        for epoch in range(1, args.max_epoch + 1):
            net.train()
            epoches.append(epoch)
            # sampler = ImbalancedSampler(train_set, num_samples=56)
            # data_loader = DataLoader(datasets=train_set, sampler=sampler, batch_size=64, num_workers=8)
            data_loader = DataLoader(dataset=train_set, batch_size=64, num_workers=8, shuffle=True)
            for i, batch_data in enumerate(data_loader):
                optimizer.zero_grad()
                X_cpu, y_cpu, infant_id_list = batch_data
                #print(infant_id_list)

                X_gpu = X_cpu.to(device, dtype=th.float)
                y_gpu = y_cpu.to(device, dtype=th.long)
                # y_gpu.shape = 64
                scores = net(x=X_gpu)
                # scores.shape = 64,2
                loss = criterion(scores, y_gpu)
                loss.backward()
                optimizer.step()
            '''if epoch == 1:
                val_roc_auc = evaluate_CM(net=net, dataset=val_set, suffix_len=args.suffix_len, device=device)
                #test_roc_auc = evaluate_CM(net=net, dataset=test_set, suffix_len=args.suffix_len, device=device)

                train_roc_auc = evaluate_CM(net=net, dataset=train_set, suffix_len=args.suffix_len, device="cuda")

                print(f"Epoch: {epoch:03}. [train] confusion_matrix: ")
                print(f"{train_roc_auc} ")
                print(f"[val] confusion_matrix: ")
                print(f"{val_roc_auc} ")
                #print(f"[test] confusion_matrix: ")
                #print(f"{test_roc_auc}")'''
            if 1:
                val_roc_auc, val_acc, val_loss,TF,FF = evaluate_CM_validation(net=net, dataset=val_set, suffix_len=args.suffix_len, device=device)
                #test_roc_auc = evaluate_CM(net=net, dataset=test_set, suffix_len=args.suffix_len, device=device)

                train_roc_auc, train_acc, train_loss = evaluate_CM(net=net, dataset=train_set, suffix_len=args.suffix_len, device="cuda")
                TFs.append(TF)
                FFs.append(FF)
                print(f"Epoch: {epoch:03}. [train] confusion_matrix: ")
                print(f"{train_roc_auc} ")
                print(f"[val] confusion_matrix: ")
                print(f"{val_roc_auc} ")
                #print(f"[test] confusion_matrix: ")
                #print(f"{test_roc_auc}")
                val_acces.append(val_acc)
                train_acces.append(train_acc)
            #if epoch % 2 == 0:
                val_roc_auc = evaluate(net=net, dataset=val_set, suffix_len=args.suffix_len, device=device)
                #test_roc_auc = evaluate(net=net, dataset=test_set, suffix_len=args.suffix_len, device=device)

                if val_roc_auc > max_val_roc_auc:
                    max_val_roc_auc = val_roc_auc
                    model_file_path = Path(args.checkpoint_dir, f'checkpoint_epoch_{epoch:03}.pth')
                    save_model(model_file_path=model_file_path, model=net)
                    mf.log_metric("max_val_roc_auc", max_val_roc_auc)

                train_roc_auc = evaluate(net=net, dataset=train_set, suffix_len=args.suffix_len, device="cuda")
                val_roc_auces.append(val_roc_auc)
                train_roc_auces.append(train_roc_auc)
                print(f"Epoch: {epoch:03}. [train] auc_score: {train_roc_auc:.4f}, acc: {train_acc:.4f}, loss: {train_loss:.4f}")
                print(f"[val] roc_auc: {val_roc_auc:.4f}, acc: {val_acc:.4f}, loss: {val_loss:.4f}")
                mf.log_metric("train_roc_auc", train_roc_auc, step=epoch)
                mf.log_metric("val_roc_auc", val_roc_auc, step=epoch)
                #mf.log_metric("test_roc_auc", test_roc_auc, step=epoch)
        epoch_df = epoch_df.append(pd.DataFrame({'epoch': epoches,
                                                 'train_acc': train_acces,
                                                 'val_acc': val_acces,
                                                 'train_roc_auc': train_roc_auces,
                                                 'val_roc_auc': val_roc_auces,
                                                 'TF': TFs,
                                                 'FF': FFs}))
        epoch_df = epoch_df[['epoch', 'train_acc', 'val_acc', 'train_roc_auc', 'val_roc_auc', 'TF', 'FF']]

        epoch_df.to_csv(Path(save_dir, f'result_{an_id:02}.csv'), index=False)


def main(an_id):
    parser = argparse.ArgumentParser(description="Parse the parameters for training the model.")
    parser.add_argument('--max-epoch', type=int, default=100, help='Max epoch.')
    parser.add_argument('--fold-id', type=int, default=0, help='fold id')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size in training and testing')
    parser.add_argument('--clip-size', type=int, default=50, help='Size of each block.')
    parser.add_argument('--stride', type=int, default=30, help='Size of each block.')
    parser.add_argument('--length', type=int, default=200, help='Length of the skeleton to use')
    parser.add_argument('--channels', type=int, default=7, help='Number of channels to use.')
    parser.add_argument('--max-hop', type=int, default=3, help='Max hop.')
    parser.add_argument('--dilation', type=int, default=1, help='Dilation.')
    parser.add_argument('--z-dim', type=int, default=128, help='The dimensionality of the GCN output features')
    parser.add_argument('--gcn-strategy', type=str, default='uniform', help='Graph spatial neighbour strategy.')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument('--suffix-len', type=int, default=0, help='Suffix length of the infant dir.')

    args = parser.parse_args()
    args.use_cuda = not args.no_cuda and th.cuda.is_available()

    conf = ConfigLoader().config
    data_root = conf['data_root']
    pose_data_root = Path(data_root, f'cooked_subsequences_0500_0100')
    fold_dir = Path(data_root, f'folds/split_absent_normal')
    #pose_data_root = Path(data_root, f'cooked_subsequences_1000_0200')
    #fold_dir = Path(data_root, f'folds/split_2class_91')
    train_label_file = Path(fold_dir, f'train_{an_id:02}.txt')
    val_label_file = Path(fold_dir, f'test_{an_id:02}.txt')


    # Use the layout with 18 joints
    args.joints = list(range(18))

    train_set = ClipDataset(data_root=pose_data_root, clip_size=args.clip_size, stride=args.stride,
                            length=args.length, label_file_path=train_label_file, suffix_len=args.suffix_len,
                            channels=args.channels)
    '''for i in range(len(train_set)):
        a,b,c=train_set[i]
        #print(c)
    print(len(train_set))
    data_loader = DataLoader(dataset=train_set, batch_size=64, num_workers=8, shuffle=True)
    for i, batch_data in enumerate(data_loader):
        X_cpu, y_cpu, infant_id_list = batch_data
        #print(infant_id_list)
        print(X_cpu.shape)
        print(y_cpu.shape)'''
    val_set = ClipDataset(data_root=pose_data_root, clip_size=args.clip_size, stride=args.stride,
                          length=args.length, label_file_path=val_label_file, suffix_len=args.suffix_len,
                          channels=args.channels)


    graph_args = dict(layout="openpose", strategy=args.gcn_strategy, max_hop=args.max_hop,
                      dilation=args.dilation)

    net = STAM(in_dim=args.channels, z_dim=args.z_dim, graph_args=graph_args, alpha_dim=128, beta_dim=128)
    # th.cuda.manual_seed(0)
    init_seed(seed=0)
    net.apply(weights_init)

    root_dir = get_root_dir()
    checkpoint_dir = Path(root_dir, 'checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    save_dir = Path(fold_dir, f'500_7_lr_{args.lr}_weight_delay_{args.weight_decay}_clip{args.clip_size}_stride_{args.stride}')
    save_dir.mkdir(parents=True, exist_ok=True)
    train(args, net, train_set, val_set, save_dir, an_id)
    return 0

if __name__ == '__main__':
    main(0)
    main(1)
    main(2)
    main(3)
    main(4)
