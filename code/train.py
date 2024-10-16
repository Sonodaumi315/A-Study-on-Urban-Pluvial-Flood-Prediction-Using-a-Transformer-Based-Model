import argparse
import json
import re
import os
from tqdm.auto import tqdm
import numpy as np
import datetime
import math
import random

import torch
from torch.optim import AdamW, Adam
from adan_pytorch import Adan
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from model_resunet import ResUNet
from model_mobilevitv2 import MobileNetViTv2

from dataset import make_dataset1
from eval import evaluate, cal

parser = argparse.ArgumentParser()
parser.add_argument('--dist', type=bool, default=False,
                help='distributed training')
parser.add_argument('--local_rank', type=int, default=-1,
                help='node rank for distributed training')

parser.add_argument('--model_load_path', type=str, default=None,
                help='model load path')
parser.add_argument('--model_save_path', type=str, default="temp",
                help='model save path')

parser.add_argument('--train_data_size', type=int, default=10000,
                help='train data size')
parser.add_argument('--valid_data_size', type=int, default=145,
                help='valid data size')
parser.add_argument('--batch_size', type=int, default=3,
                help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=3,
                help='eval batch size')
parser.add_argument('--accumulation', type=int, default=1,
                help='gradient accumulation steps')
parser.add_argument('--max_lr', type=float, default=1e-3,#1e-5,
                help='max learning rate')
parser.add_argument('--min_lr', type=float, default=5e-5,#5e-7,
                help='min learning rate')
parser.add_argument('--max_norm', type=float, default=1.0,
                help=" clipping gradient norm")
parser.add_argument('--epochs', type=int, default=100,
                help='training epochs')
parser.add_argument('--in_channels', type=int, default=6,
                help='input channels (please don\'t modify)')
parser.add_argument('--valid_interval', type=int, default=1,
                help='the interval between 2 validations')
parser.add_argument('--valid_per_epoch', type=int, default=5,
                    help='valid per epoch (when valid_interval=1)')
parser.add_argument('--save_interval', type=int, default=5,
                help='the interval between 2 saved models')

parser.add_argument('--schedule_var1', type=int, default=0,
                help='schedule var1')
parser.add_argument('--schedule_var2', type=int, default=0,
                help='schedule var2')
parser.add_argument('--seed', type=int, default=3407,#233,
                help='random seed')
parser.add_argument('--wd', type=float, default=1e-2,
                help='weight decay')

parser.add_argument('--optimizer_class', type=int, default=0,
                help='type of optimizer  0:adamw   1:adan')
parser.add_argument('--schedule', type=int, default=0,
                help='type of schedule  0:cosine   1:linear   2:cycle')
parser.add_argument('--lr_step_interval', type=int, default=0,
                help='0: per epoch   1: per batch')
parser.add_argument('--augs', type=bool, default=False,
                help='use data augmentation for spatial terrain inputs')
parser.add_argument('--augr', type=int, default=0,
                help='use data augmentation for rainfall inputs (num of synthetic samples,  augr=2 means that 4 extra synthetic sampels, 2 more heavier rainfalls and 2 more lighter rainfalls)')
parser.add_argument('--delta_p', type=float, default=0.01,
                help='the maximum proportion of change in total rainfall (please don\'t modify)')  

def get_num(v):
    if v<10:
        a="00%d"%(v)
    elif v<100:
        a="0%d"%(v)
    else:
        a="%d"%(v)
    return a

def ts2fl(v, model_save_path):
    f=open("../models/%s/temp.txt"%(model_save_path),"w")
    f.write(" %.8lf "%(v))
    f.close()
    f=open("../models/%s/temp.txt"%(model_save_path),"r")
    q = f.readline()
    q = re.findall(" ([\s\S]*?) ",q,re.M)
    v = float(q[0])
    f.close()
    return v

def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def validate(valid_dataset, model, local_rank, augr = False):
    
    if args.dist:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda")
        
    model.eval()

    l_ACC_005 = 0; l_ACC_030 = 0
    l_H_005   = 0; l_H_030   = 0
    l_M_005   = 0; l_M_030   = 0
    l_FP_005  = 0; l_FP_030  = 0
    l_TOTAL   = 0
    flSSE     = 0
    flpix     = 0
    flSSE0    = 0
    tot_loss_valid = 0
    tot_valid      = 0
    tot_sample     = 0
    print_cnt = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataset):

            print_cnt, tot_loss_valid, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = evaluate(
                model, batch[0], batch[1], batch[2], batch[3], True, print_cnt, tot_loss_valid, 
                flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL,
                local_rank)
            
            tot_valid += 1   
            tot_sample += batch[0].shape[0]
        
    l_ACC_005 = torch.tensor(l_ACC_005, dtype=float).to(device); l_ACC_030 = torch.tensor(l_ACC_030, dtype=float).to(device)
    l_H_005   = torch.tensor(l_H_005, dtype=float).to(device);   l_H_030   = torch.tensor(l_H_030, dtype=float).to(device)
    l_M_005   = torch.tensor(l_M_005, dtype=float).to(device);   l_M_030   = torch.tensor(l_M_030, dtype=float).to(device)
    l_FP_005  = torch.tensor(l_FP_005, dtype=float).to(device);  l_FP_030  = torch.tensor(l_FP_030, dtype=float).to(device)
    l_TOTAL   = torch.tensor(l_TOTAL, dtype=float).to(device)
    flSSE     = torch.tensor(flSSE, dtype=float).to(device)
    flpix     = torch.tensor(flpix, dtype=float).to(device)
    flSSE0    = torch.tensor(flSSE0, dtype=float).to(device)
    tot_loss_valid = torch.tensor(tot_loss_valid, dtype=float).to(device)
    tot_valid      = torch.tensor(tot_valid, dtype=float).to(device)
    tot_sample     = torch.tensor(tot_sample, dtype=int).to(device)
    if args.dist:
        l_ACC_005 = reduce_sum(l_ACC_005); l_ACC_030 = reduce_sum(l_ACC_030)
        l_H_005   = reduce_sum(l_H_005);   l_H_030   = reduce_sum(l_H_030)
        l_M_005   = reduce_sum(l_M_005);   l_M_030   = reduce_sum(l_M_030)
        l_FP_005  = reduce_sum(l_FP_005);  l_FP_030  = reduce_sum(l_FP_030)
        l_TOTAL   = reduce_sum(l_TOTAL)
        flSSE     = reduce_sum(flSSE)
        flpix     = reduce_sum(flpix)
        flSSE0    = reduce_sum(flSSE0)
        tot_loss_valid = reduce_sum(tot_loss_valid)
        tot_valid      = reduce_sum(tot_valid)
        tot_sample     = reduce_sum(tot_sample)
        dist.barrier()
    tot_loss_valid/=tot_valid
    if (args.dist == False) or (local_rank == 0):
        print("       valid tot_sample ", tot_sample)
        flMSE = cal("valid", tot_loss_valid, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, False)
    else:
        flMSE = None

    return tot_loss_valid, flMSE
    

def main_worker(local_rank, nprocs, args, model_class):
    print(" start self supervised learning  local_rank:%d  nprocs:%d"%(local_rank, nprocs))
    
    if args.dist:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl')
    else:
        device = torch.device("cuda")
    cudnn.benchmark = True
    
    #get parameters 
    num_epochs      = args.epochs
    batch_size      = args.batch_size
    eval_batch_size = args.eval_batch_size
    accumulation    = args.accumulation
    valid_interval  = args.valid_interval
    save_interval   = args.save_interval
    
    model_save_path = args.model_save_path
    model_save_path = "%s %s"%(str(datetime.date.today()),str(datetime.datetime.now().time())[:8].replace(":","-"))
    model_load_path = args.model_load_path
    
    train_losses = []
    valid_losses = []
    #create model
    
    load_epoch = 0
    if not model_load_path:
        model = model_class()
        load_epoch = 0
    else:
        #model = torch.load(model_load_path)
        model = model_class()
        model.load_state_dict(torch.load(model_load_path))
    
    if args.dist:
        print(" model convert_sync_batchnorm ")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model.to(device)
    
    if args.dist:
        args.batch_size      = args.batch_size // nprocs
        args.eval_batch_size = args.eval_batch_size // nprocs
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])#, find_unused_parameters=True)
        #model = model.module
    
    #get dataset
    if args.dist:
        train_dataset, train_steps, train_sampler, valid_dataset, valid_steps, valid_sampler = make_dataset1(batch_size = args.batch_size, train_data_size = args.train_data_size, eval_batch_size = args.eval_batch_size, valid_data_size = args.valid_data_size, dist = args.dist, nprocs = nprocs, augs = args.augs, augr = args.augr, save_path = None, delta_p = args.delta_p)
    else:
        train_dataset, train_steps, valid_dataset, valid_steps = make_dataset1(batch_size = args.batch_size, train_data_size = args.train_data_size, eval_batch_size = args.eval_batch_size, valid_data_size = args.valid_data_size, dist = args.dist, nprocs = nprocs, augs = args.augs, augr = args.augr, save_path = None, delta_p = args.delta_p)
    num_train_data     = len(train_dataset)
    num_valid_data     = len(valid_dataset)
    train_eval_num  = max(1, 200 // batch_size)
    
    if args.optimizer_class == 0:
        optimizer_class = AdamW
        if (args.dist == False) or (local_rank == 0):
            print("   use Adamw as optimizer")
    elif args.optimizer_class == 1:
        optimizer_class = Adan
        if (args.dist == False) or (local_rank == 0):
            print("   use Adan as optimizer")
    
    if args.schedule == 0:
        print("---------use CosineAnnealingLR")
        optimizer = optimizer_class(filter(lambda p:p.requires_grad, model.parameters()), lr=args.max_lr, weight_decay=args.wd)
        if args.lr_step_interval == 0:
            lr_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.schedule_var1,                eta_min = args.min_lr)
        else:
            lr_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.schedule_var1*num_train_data, eta_min = args.min_lr)
    elif args.schedule == 1: 
        print("---------use linear_schedule")
        optimizer = optimizer_class(filter(lambda p:p.requires_grad, model.parameters()), lr=1, weight_decay=args.wd)
        def linear_schedule(up_lr=5e-3, lo_lr=1e-5, schedule_var1 = 0, schedule_var2 = 0, num_epochs = 250):
            def schedule(epoch):
                if epoch < schedule_var1:
                    lr = up_lr
                elif epoch < schedule_var2:
                    lr = up_lr - (up_lr-lo_lr)*(epoch-schedule_var1)/(schedule_var2-schedule_var1)
                else:
                    lr = lo_lr
                return (lr)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule, last_epoch=-1)
        if args.lr_step_interval == 0:
            lr_scheduler = linear_schedule(up_lr = args.max_lr, lo_lr = args.min_lr, schedule_var1 = args.schedule_var1, 
                                           schedule_var2 = args.schedule_var2,                num_epochs = num_epochs)
        else:
            lr_scheduler = linear_schedule(up_lr = args.max_lr, lo_lr = args.min_lr, schedule_var1 = args.schedule_var1*num_train_data, 
                                           schedule_var2 = args.schedule_var2*num_train_data, num_epochs = num_epochs*num_train_data)
    elif args.schedule == 2:
        print("---------use cycle_schedule")
        optimizer = optimizer_class(filter(lambda p:p.requires_grad, model.parameters()), lr=1, weight_decay=args.wd)
        #optimizer = Adam(model.parameters(), lr = 1, betas = (0.9,0.999), eps = 1e-7, amsgrad=False)
        def step_decay_schedule(up_lr=5e-3, lo_lr=1e-4, decay_factor=0.9, stepsize=5, work_epochs = 500, basepoch=0):
            def schedule(epoch):
                epoch2=epoch+basepoch
                if epoch2<work_epochs:
                    cycle=np.floor(1+epoch2/(2*stepsize))
                    lrf=1-np.abs(epoch2/stepsize-2*cycle+1)
                    lr=(up_lr-lo_lr)*lrf*np.power(decay_factor,epoch2/stepsize)+lo_lr
                else:
                    lr=lo_lr
                return (lr)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule, last_epoch=-1)
        if args.lr_step_interval == 0:
            lr_scheduler = step_decay_schedule(up_lr = args.max_lr, lo_lr = args.min_lr, decay_factor=0.95, stepsize=10, 
                                               work_epochs=num_epochs)
        else:
            lr_scheduler = step_decay_schedule(up_lr = args.max_lr, lo_lr = args.min_lr, decay_factor=0.95, stepsize=10*num_train_data, 
                                               work_epochs=num_epochs*num_train_data)
    
    criterion = torch.nn.MSELoss()
    weight = 1
    if args.augs == True:
        train_eval_num = train_eval_num*8
        
    if (args.dist == False) or (local_rank == 0):
        print(" weight: %.3lf   target:%s"%(weight, "labels"))
        print(" num_train_data: %d"%(num_train_data))
        print(" num_valid_data: %d"%(num_valid_data))
        print(" train_steps: %d"%(train_steps))
        print(" valid_steps: %d"%(valid_steps))
        
        print(" batch_size: %d"%(batch_size))
        print(" eval_batch_size: %d"%(eval_batch_size))
        print(" accumulation: %d"%(accumulation))
        print(" model_class: %s"%(str(model_class)))
        print(" optimizer_class: %d"%(args.optimizer_class))
        print(" augs: %d"%(args.augs))
        print(" augr: %d"%(args.augr))
        print(" max_lr: %.8lf"%(args.max_lr))
        print(" min_lr: %.8lf"%(args.min_lr))
        print(" weight decay: ", args.wd)
        print(" schedule: ", args.schedule)
        print(" lr_step_interval: %d"%(args.lr_step_interval))
        print(" epochs: %d"%(num_epochs))
        print(" schedule_var1: %d"%(args.schedule_var1))
        print(" schedule_var2: %d"%(args.schedule_var2))
        
        print(" model save path: %s"%(model_save_path))
        print(" model load path: %s"%(model_load_path))
        print(" load_epoch: %s"%(load_epoch))
        
        print(" train_eval_num: %d"%(train_eval_num))
        
        if not os.path.exists("../models/%s"%(model_save_path)):
            os.makedirs("../models/%s"%(model_save_path))
            
    valid_batch = []
    if args.valid_interval == 1:
        a = num_train_data // args.valid_per_epoch
        for i in range(1, args.valid_per_epoch):
            valid_batch.append(int(i*a))
    print(" valid_batch ", valid_batch)
    
    best_model = 1e9
    
    def eval_model(best_model, cnt_valid):
        if epoch > load_epoch:
            valid_loss, flMSE = validate(valid_dataset, model, local_rank, args.augr)
        else:
            valid_loss = valid_losses[int(epoch/valid_interval)-1]
        
        if valid_loss < best_model:
            best_model = valid_loss
            if ((args.dist == False) or (local_rank == 0)) and (epoch > load_epoch):
                if args.dist == False:
                    save_model = model.state_dict()
                else:
                    save_model = model.module.state_dict()
                torch.save(save_model, "../models/%s/best_model.bin"%(model_save_path))
        if ((args.dist == False) or (local_rank == 0)) and (epoch > load_epoch):
            a = get_num(epoch)
            if args.dist == False:
                save_model = model.state_dict()
            else:
                save_model = model.module.state_dict()
            torch.save(save_model, "../models/%s/ep-%s-%d.bin"%(model_save_path,a,cnt_valid))    

        if (args.dist == False) or (local_rank == 0):
            print("---epoch %d/%d  valid_loss:%.5lf  best_model_loss:%.5lf"%(epoch, num_epochs, valid_loss, best_model))
            
            valid_losses.append(ts2fl(valid_loss, model_save_path))
            with open("../models/%s/valid_losses.json"%(model_save_path),"w",encoding='utf-8') as fout:
                json.dump(valid_losses,fout,indent=4,ensure_ascii=False)  
        
        return best_model
    
    for epoch in tqdm(range(1,num_epochs+1)):
        if args.dist:
            train_sampler.set_epoch(epoch+1)
        
        model.train()
        cnt_valid = 0
        
        l_ACC_005 = 0; l_ACC_030 = 0
        l_H_005   = 0; l_H_030   = 0
        l_M_005   = 0; l_M_030   = 0
        l_FP_005  = 0; l_FP_030  = 0
        l_TOTAL   = 0
        flSSE     = 0
        flpix     = 0
        flSSE0    = 0
        tot_loss_train = 0
        tot_train      = 0
        print_cnt = 0
        
        tot_phy_loss = 0
        #train_time1 = time.time()
        for batch in tqdm(train_dataset):
            if epoch > load_epoch:
                inputs = batch[0].permute(0, 3, 1, 2).contiguous()
                if len(batch[1].shape) == 3:
                    rain_p = batch[1][:, 0, :].to(device, non_blocking=True)
                else:
                    rain_p = batch[1].to(device, non_blocking=True)
                labels = batch[2].to(device, non_blocking=True)
                
                inputs = inputs.to(device, non_blocking=True)
                output = model(inputs = inputs, rain_p = rain_p, labels = labels)
                loss = output["loss"] / float(accumulation)
                
                tot_train += 1
                tot_loss_train += loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                
                if tot_train <= train_eval_num:
                    print_cnt, tot_loss_train, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = evaluate(
                        None, output["results"], None, batch[2], None, True, print_cnt, tot_loss_train, 
                        flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL,
                        local_rank)

                if tot_train % accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                target = labels.view(output["results"].shape)
                phyloss = 0   
                for i in range(args.augr):
                    rain_p = batch[1][:, 1+i, :].to(device)
                    output = model(inputs = inputs, rain_p = rain_p)
                    results_increase = output["results"]
                    target_increase = torch.max(target, results_increase).to(device, non_blocking=True)
                    loss = criterion(results_increase, target_increase)*weight

                    phyloss += loss
                    loss.backward()
                    if tot_train % accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    rain_p = batch[1][:, 1+args.augr+i, :].to(device)
                    output = model(inputs = inputs, rain_p = rain_p)
                    results_decrease = output["results"]
                    target_decrease = torch.min(target, results_decrease).to(device, non_blocking=True)
                    loss = criterion(results_decrease, target_decrease)*weight

                    phyloss += loss
                    loss.backward()
                    if tot_train % accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        
                if args.augr > 0:
                    phyloss /= args.augr*2*weight
                tot_phy_loss += phyloss
                
            if tot_train in valid_batch:
                cnt_valid += 1
                best_model = eval_model(best_model, cnt_valid)
                model.train()
            if args.lr_step_interval == 1: 
                lr_scheduler.step()

        if args.lr_step_interval == 0: 
            lr_scheduler.step()
        
        r''''''
        if args.dist:
            l_ACC_005 = torch.tensor(l_ACC_005, dtype=float).to(device); l_ACC_030 = torch.tensor(l_ACC_030, dtype=float).to(device)
            l_H_005   = torch.tensor(l_H_005, dtype=float).to(device);   l_H_030   = torch.tensor(l_H_030, dtype=float).to(device)
            l_M_005   = torch.tensor(l_M_005, dtype=float).to(device);   l_M_030   = torch.tensor(l_M_030, dtype=float).to(device)
            l_FP_005  = torch.tensor(l_FP_005, dtype=float).to(device);  l_FP_030  = torch.tensor(l_FP_030, dtype=float).to(device)
            l_TOTAL   = torch.tensor(l_TOTAL, dtype=float).to(device)
            flSSE     = torch.tensor(flSSE, dtype=float).to(device)
            flpix     = torch.tensor(flpix, dtype=float).to(device)
            flSSE0    = torch.tensor(flSSE0, dtype=float).to(device)
            tot_loss_train = torch.tensor(tot_loss_train, dtype=float).to(device)
            tot_train      = torch.tensor(tot_train, dtype=float).to(device)
            tot_phy_loss   = torch.tensor(tot_phy_loss, dtype=float).to(device)
            l_ACC_005 = reduce_sum(l_ACC_005); l_ACC_030 = reduce_sum(l_ACC_030)
            l_H_005   = reduce_sum(l_H_005);   l_H_030   = reduce_sum(l_H_030)
            l_M_005   = reduce_sum(l_M_005);   l_M_030   = reduce_sum(l_M_030)
            l_FP_005  = reduce_sum(l_FP_005);  l_FP_030  = reduce_sum(l_FP_030)
            l_TOTAL   = reduce_sum(l_TOTAL)
            flSSE     = reduce_sum(flSSE)
            flpix     = reduce_sum(flpix)
            flSSE0    = reduce_sum(flSSE0)
            tot_loss_train = reduce_sum(tot_loss_train)
            tot_train      = reduce_sum(tot_train)
            tot_phy_loss   = reduce_sum(tot_phy_loss)
            dist.barrier()
        
        if epoch > load_epoch:
            tot_loss_train/=tot_train
            if (args.dist == False) or (local_rank == 0):
                train_losses.append(ts2fl(tot_loss_train, model_save_path))
        else:
            tot_loss_train = train_losses[epoch-1]
        
        if epoch%valid_interval==0:
            cnt_valid += 1
            best_model = eval_model(best_model, cnt_valid)
        
        if (args.dist == False) or (local_rank == 0):
            print(" train_evaluate %d*%d"%(train_eval_num, batch_size))
            print("   tot_phy_loss: %.6lf"%(tot_phy_loss/tot_train))
            cal("train", tot_loss_train, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, False)
            print("###epoch %d/%d  train_loss:%.5lf/%.5lf  lr=%.8lf"%(epoch, num_epochs, tot_loss_train, math.sqrt(tot_loss_train), optimizer.param_groups[0]["lr"]))
            
            if epoch%save_interval==0:
                if epoch > load_epoch:
                    a = get_num(epoch)
                    if args.dist == False:
                        save_model = model.state_dict()
                    else:
                        save_model = model.module.state_dict()
                    torch.save(save_model, "../models/%s/ep-%s_%.3lf.bin"%(model_save_path,a,tot_loss_train))
                
                with open("../models/%s/train_losses.json"%(model_save_path),"w",encoding='utf-8') as fout:
                    json.dump(train_losses,fout,indent=4,ensure_ascii=False)
        
def setup_seed(seed):
    print(" seed:%d"%(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.deterministic = False

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

if __name__=="__main__":
    
    args = parser.parse_args()
    args.seed = 666
    
    args.max_lr = 1e-3
    args.min_lr = 1e-5
    args.epochs = 100
    args.schedule_var1 = 5
    args.schedule_var2 = 101
    args.wd = 1e-2
    args.optimizer_class = 0
    args.schedule = 0
    args.augs   = True
    args.augr  = 0
    args.delta_p = 0.01
    args.lr_step_interval = 0
    args.valid_per_epoch = 8
    args.valid_interval = 1
    args.save_interval = 1
    
    args.train_data_size = 10000
    args.batch_size   = int(32)
    args.eval_batch_size   = int(32)
    args.accumulation = 1
    model_class  = MobileNetViTv2
    use_gpu      = True
    
    args.dist = (args.local_rank != -1)
    args.nprocs = torch.cuda.device_count()
    if args.seed is not None:
        setup_seed(args.seed)
    main_worker(args.local_rank, args.nprocs, args, model_class)   