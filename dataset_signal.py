import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import pandas as pd

def  fsl_split(total_x, total_y, classes_num, split_index, sample, path, **kwargs):
    save_path = 'dataset/'+path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    arr = np.arange(classes_num)
    np.random.shuffle(arr)
    sub_arrs = np.split(arr, [split_index])
    rand_indices = torch.randperm(sample)
    temp_x, temp_y = [], []
    temp_ft_x,  temp_ft_y = [], []
    temp_ts_x,  temp_ts_y = [], []
    for i in sub_arrs[0]:
        index = torch.where(total_y == i)
        index_rand = rand_indices[0:1000] + index[0][0]
        val_ft_rand = rand_indices[0:100] + index[0][0]
        val_ts_rand = rand_indices[500:] + index[0][0]
        temp_x.append(total_x[index_rand])
        temp_y.append(total_y[index_rand])
        temp_ft_x.append(total_x[val_ft_rand])
        temp_ft_y.append(total_y[val_ft_rand])
        temp_ts_x.append(total_x[val_ts_rand])
        temp_ts_y.append(total_y[val_ts_rand])
    train_x = torch.vstack(temp_x)
    train_y = torch.hstack(temp_y)
    val_ft_x = torch.vstack(temp_ft_x)
    val_ft_y = torch.hstack(temp_ft_y)
    val_ts_x = torch.vstack(temp_ts_x)
    val_ts_y = torch.hstack(temp_ts_y)
    torch.save(train_x, save_path+'/train_x.csv')
    torch.save(train_y, save_path+'/train_y.csv')
    temp_1_x, temp_1_y = [], []
    temp_5_x, temp_5_y = [], []
    temp_x, temp_y = [], []
    for i in sub_arrs[1]:
        index = torch.where(total_y == i)
        index_rand = rand_indices[0:1] + index[0][0]
        temp_1_x.append(total_x[index_rand])
        temp_1_y.append(total_y[index_rand])
        index_rand = rand_indices[0:5] + index[0][0]
        temp_5_x.append(total_x[index_rand])
        temp_5_y.append(total_y[index_rand])
        index_rand = rand_indices[-500:] + index[0][0]
        temp_x.append(total_x[index_rand])
        temp_y.append(total_y[index_rand])

    query_1_x = torch.vstack(temp_1_x)
    query_1_y = torch.hstack(temp_1_y)
    query_5_x = torch.vstack(temp_5_x)
    query_5_y = torch.hstack(temp_5_y)
    test_x = torch.vstack(temp_x)
    test_y = torch.hstack(temp_y)
    torch.save(query_1_x, save_path+'/query_x_1.csv')
    torch.save(query_1_y, save_path+'/query_y_1.csv')
    torch.save(query_5_x, save_path+'/query_x_5.csv')
    torch.save(query_5_y, save_path+'/query_y_5.csv')
    torch.save(test_x, save_path+'/test_x.csv')
    torch.save(test_y, save_path+'/test_y.csv')
    val_x = [val_ft_x,val_ts_x]
    val_y = [val_ft_y,val_ts_y]
    return train_x,train_y,val_x,val_y,query_5_x,query_5_y,test_x,test_y,kwargs["classes"]

def RML_data(path,snr):
    if 'hdf5' in path:
        with h5py.File(path, 'r') as f:
            print(f.keys())
            snr_list = f['Z'][:].squeeze()
            index_list = np.where(snr_list == snr)
            total_x = torch.Tensor(f['X'][index_list].transpose(0,2,1))
            label = torch.Tensor(f['Y'][index_list])
            _, total_y = torch.where(label == 1)
            return fsl_split(total_x,total_y,24,14, sample=sum(total_y==0),path='10a_Temp',)
    else:
        data = pd.read_pickle(path)
        snr = snr
        x,y =[],[]
        classes = []
        snr_list = []
        for k, v in data.items():
            if k[0] not in classes:
                classes.append(k[0])
            if k[1] not in snr_list:
                snr_list.append(k[1])
            if k[1] == snr:
                x.append(torch.Tensor(v))
                y.append(torch.full((v.shape[0],), classes.index(k[0])))
        snr_list.sort()
        total_x = torch.vstack(x)
        total_y = torch.hstack(y)
        return fsl_split(total_x, total_y, len(classes), 6, sample=sum(total_y==0), path='10a_Temp', classes=classes)

def resort_label(labels, classes=None):
    unique_classes = torch.unique(labels)
    class_mapping = {class_item.item(): i for i, class_item in enumerate(unique_classes)}
    remapped_data = torch.tensor([class_mapping[item.item()] for item in labels])
    if classes:
        new_classes = [classes[i] for i in unique_classes]
    else:
        new_classes = [str(x) for x in range(0, len(unique_classes))]
    return remapped_data, new_classes

def dataset_RML(args):
    args.dataset = 'RML'
    args.ft_ratio = '5shot'
    train_x,train_y,val_x,val_y,query_x,query_y,test_x,test_y,classes = RML_data('./dataset/RML2016_10a.pkl',args.snr)
    query_y, query_classes = resort_label(query_y, classes)
    test_y, test_classes = resort_label(test_y, classes)

    train_dataset = TensorDataset(train_x,train_y)
    query_dataset = TensorDataset(query_x, query_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_dataset.classes = classes
    query_dataset.classes = query_classes
    test_dataset.classes = test_classes

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=True
    )

    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=True,
    )

    return train_loader,query_loader,test_loader

