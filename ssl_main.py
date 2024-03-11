import torch
import time
import argparse
import os
import torch.optim as optim
from model.models import get_model
import numpy as np
from unspervise_learning.utils.data_utils import augmentation, AverageMeter, calculate_top_k_accuracy
from dataset_signal import dataset_RML

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=53)  # 54 and 53
parser.add_argument('--cuda', type=int, default=0)
# Training argument
parser.add_argument('--pre_train', type=bool, default= True,
                    help='If True, config for contrastive training mode')
parser.add_argument('--pr_epochs', type=int, default=300,
                    help='Total pre-training epochs for the framework')
parser.add_argument('--pr_lr', type=float, default=0.001,
                    help='Pre-training learning rate of the optimizer')
parser.add_argument('--ev_epochs', type=int, default=2000,
                    help='Total evaluation-training epochs for the framework')
parser.add_argument('--ev_lr', type=float, default=0.0001,
                    help='Pre-training learning rate of the optimizer')
# Logging argument
parser.add_argument('--log_interval', type=int, default=240)
# Network argument
parser.add_argument('--framework', type=str, default='lsm',
                    choices=['lsm'],
                    help='name of framework')
parser.add_argument('--backbone', type=str, default='xciT',
                    choices=['efficient_net_b0',
                             'efficient_net_b2', 'efficient_net_b4','xciT'],
                    help='name of backbone network')
parser.add_argument('--emb_size', type=int, default=1344,
                    help='embedding size of the backbone')
parser.add_argument('--in_channels', type=int, default=2,
                    help='input channels')



# Common dataset argument
parser.add_argument('--batch_size', type=int, default=1400,
                    help='batch size of the loading dataset')
parser.add_argument('--patch_len', type=int, default=128 // 10,
                    help='patch len of the data')



def model_saving_config(args):
    model_root = './Pre_training_pt'
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    dataset_root = model_root + '/' + args.dataset
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)
    model_name = dataset_root + '/' + args.framework + '_' + args.backbone + '_' + \
        str(args.batch_size) + '_' + str(args.patch_len) + \
        '_' + str(args.pr_epochs)
    return model_name

def pre_train(args, train_dataloader):
    model = get_model(args.backbone, args.patch_len,
                      args.emb_size, args.in_channels, args.classes, backbone_pretrain=False)

    if torch.cuda.device_count() > 1:
        print('Available GPUs: ' + str(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.pr_lr)
    loss_list = []
    for epoch in range(args.pr_epochs):
        model.train()
        avg_loss = AverageMeter()
        min_loss = 100000000
        save_path = None
        for batch_idx, (data, _) in enumerate(train_dataloader):
            data_t = augmentation(data, args.patch_len, patch=True, ration=True, overturn=False, flip=False).to(args.device)
            data_t_a = augmentation(data, args.patch_len, patch=True, ration=True, overturn=False, flip=False).to(args.device)
            if torch.cuda.device_count() > 1:
                loss = model(data_t, data_t_a)
                loss = loss.mean()
            else:
                loss = model(data_t, data_t_a)
            avg_loss.update(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(avg_loss.avg.item())
        if epoch % 100 == 0:
            print("epoch:", epoch, "done")
        if avg_loss.avg < min_loss :
            if save_path != None:
                os.remove(save_path)
            save_path = args.model_name + '_'+str(args.snr)+'_min_loss.pth'
            torch.save(model.state_dict(), save_path)
    args.model_name = args.model_name + '_'+str(args.snr)+'_min_loss.pth'
    print('Save model success')

def fine_tuning_and_test(args, train_dataloader, test_dataloader):
    model = get_model(args.backbone, args.patch_len,
                      args.emb_size, args.in_channels, args.classes,
                      backbone_pretrain=False).to(args.device)

    state_dict = torch.load(args.model_name)
    model.load_state_dict(state_dict, strict=False)
    model_param_sum = 0
    for n, p in model.named_parameters():
        if 'cla_head' not in n:
            p.requires_grad = False
        if p.requires_grad == True:
            model_param_sum += 1
    print('model_param_sum:', model_param_sum)

    if torch.cuda.device_count() > 1:
        print('Available GPUs: ' + str(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.ev_lr)

    log_path = './log/ft_'+args.backbone+'_'+args.framework+'_'+args.dataset+'_'+str(args.snr)+'_'+args.ft_ratio+'.txt'

    temp_ckpt_file = None
    best_test_acc_top1 = 0
    best_test_acc_top5 = 0
    best_train_acc = 0
    ft_loss_list = []

    for epoch in range(args.ev_epochs):
        model.train()
        loss_A = AverageMeter()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            data_t = augmentation(data, args.patch_len, patch=True, ration=True, overturn=False, flip=False).to(args.device)

            target = target.to(args.device)
            if torch.cuda.device_count() > 1:
                loss = model.module.train_one_cla_step(data_t, target, criterion)
                loss.mean().backward()
                loss_A.update(loss.mean())
            else:
                loss = model.train_one_cla_step(data_t, target, criterion)
                loss.backward()
                loss_A.update(loss)
            optimizer.step()
        ft_loss_list.append(loss_A.avg.item())

        if epoch % 100 == 0:
            model.eval()
            correct = 0
            correct1 = 0
            y_predict =[]
            y_ture = []
            features = []
            labels = []
            train_output_logits = []
            test_output_logits = []
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_dataloader):
                    data = data.float().to(args.device)
                    target = target.to(args.device)
                    if torch.cuda.device_count() > 1:
                        feature, output = model.module.predict(data)
                    else:
                        feature, output = model.predict(data)
                    features.append(feature)
                    labels.append(target)
                    test_output_logits.append(output)
                    pred = output.data.max(1, keepdim=True)[1]
                    y_predict.extend(pred.squeeze().tolist())
                    y_ture.extend(target.tolist())
                    correct += pred.eq(target.data.view_as(pred)).sum().item()
                labels = torch.hstack(labels)
                test_top_1 = calculate_top_k_accuracy(torch.vstack(test_output_logits).cpu().numpy(), labels.cpu().numpy(), 1) * 100
                test_top_5 = calculate_top_k_accuracy(torch.vstack(test_output_logits).cpu().numpy(), labels.cpu().numpy(), 5) * 100
                for batch_idx, (data1, target1) in enumerate(train_dataloader):
                    data1 = data1.float().to(args.device)
                    target1 = target1.to(args.device)
                    if torch.cuda.device_count() > 1:
                        _, output = model.module.predict(data1)
                    else:
                        _, output = model.predict(data1)
                    train_output_logits.append(output)
                    pred1 = output.data.max(1, keepdim=True)[1]
                    correct1 += pred1.eq(target1.data.view_as(pred1)).sum().item()
            train_acc = 100. * correct1 / len(train_dataloader.dataset)
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if test_top_1 > best_test_acc_top1:
                best_test_acc_top1 = test_top_1
                if temp_ckpt_file != None:
                    os.remove(temp_ckpt_file)
                temp_ckpt_file = args.model_name + '_ft_'+ args.ft_ratio + '_' + str(round(best_test_acc_top1,2)) + '.pth'
                torch.save(model.state_dict(), temp_ckpt_file)
            if test_top_5 > best_test_acc_top5:
                best_test_acc_top5 = test_top_5
            if not os.path.exists(log_path):
                with open(log_path, "w") as file:
                    file.write("")  # 写入内容或留空
            log = open(log_path, 'a')
            log.write('\n model_layer_sum:%d [epoch %d] train_loss_avg: %.3f train_acc: %.3f test_acc_top1: %.3f test_acc_top5: %.3f best_test_acc_top1:%.3f best_test_acc_top5:%.3f'%
                      (model_param_sum ,epoch + 1, round(loss_A.avg.item(), 2),train_acc, test_top_1, test_top_5, best_test_acc_top1, best_test_acc_top5))
            log.close()

    return best_test_acc_top1

def set_seed(seed=666):
    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.patch_len = 128 // 10
    args.classes = 5
    args.snr = 8
    set_seed(args.seed)
    st_time = time.time()

    train_dataloader, query_dataloader, test_dataloader = dataset_RML(args)
    for train_flag in [False, False]:
        args.pre_train = train_flag
        if args.pre_train:
            args.model_name = model_saving_config(args)
            pre_train(args, train_dataloader)
        else:
            args.model_name = './Pre_training_pt/RML/lsm_xciT_1400_12_300_8_min_loss.pth'
            print("Fine-tuning samples:",len(query_dataloader.dataset), "; Test samples:",len(test_dataloader.dataset))
            acc = fine_tuning_and_test(args, query_dataloader, test_dataloader)
            print('Snr = ',args.snr,' Final accuracy: ', acc)
            if not os.path.exists("./log/result"+args.ft_ratio+".txt"):
                with open("./log/result"+args.ft_ratio+".txt", "w") as file:
                    file.write("")
            log = open("./log/result_"+str(args.seed)+"_"+args.ft_ratio+".txt", 'a')
            log.write('Snr = %d  ACC:%.3f \n ' %(args.snr, acc))
            log.close()
