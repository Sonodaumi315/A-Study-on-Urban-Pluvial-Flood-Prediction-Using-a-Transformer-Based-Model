import json
import os
from tqdm.auto import tqdm
import numpy as np
import math
import torch
from model_resunet import ResUNet
from model_mobilevitv2 import MobileNetViTv2

from dataset import UfloodDataset, make_dataset1, make_dataset2

def get_num(v):
    if v<10:
        a="00%d"%(v)
    elif v<100:
        a="0%d"%(v)
    else:
        a="%d"%(v)
    return a

def div(a, b):
    if b == 0:
        return 0
    return a/b
    
def cal(prefix, tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, print_loss = True):
    
    flMSE = div(flSSE, flpix)
    flMSE0= div(flSSE0, flpix)
    csi05 = div(l_H_005, (l_H_005+l_M_005+l_FP_005))
    csi30 = div(l_H_030, (l_H_030+l_M_030+l_FP_030))
    p05   = div(l_H_005, (l_H_005+l_FP_005))
    p30   = div(l_H_030, (l_H_030+l_FP_030))
    r05   = div(l_H_005, (l_H_005+l_M_005))
    r30   = div(l_H_030, (l_H_030+l_M_030))
    f05   = div(2*p05*r05, (p05+r05))
    f30   = div(2*p30*r30, (p30+r30))
    
    if print_loss:
        print("   tot_loss:%.6lf"%(tot_loss))
    print("   flMSE:%.6lf=%.6lf/%d flMSE0:%.6lf=%.6lf/%d    %s-RMSE:%.6lf RMSE0:%.6lf "%(flMSE, flSSE, flpix, flMSE0, flSSE0, flpix, prefix, math.sqrt(flMSE), math.sqrt(flMSE0)))
    print("   030cases %d %d %d   Ann/Ahd=%.3lf    %s-acc_005:%.6lf %s-acc_030:%.6lf %s-csi_005:%.6lf %s-csi_030:%.6lf  %s-fs_005:%.6lf %s-fs_030:%.6lf"%(l_H_030, l_M_030, l_FP_030, div(l_H_005+l_FP_005, l_H_005+l_M_005),prefix, div(l_ACC_005, l_TOTAL), prefix, div(l_ACC_030, l_TOTAL), prefix, csi05, prefix, csi30, prefix, f05, prefix, f30))
    return flMSE

def evaluate(model, testx, testr, testy, testm, istrain, print_cnt, tot_loss,
                flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL,
                local_rank):
    cpu_device = torch.device("cpu")
    with torch.no_grad():
        if model is not None:
            
            if local_rank == -1:
                device = torch.device("cuda")
            else:
                device = torch.device("cuda", local_rank)
                
            if torch.is_tensor(testy) == True:
                inputs = testx.permute(0, 3, 1, 2).contiguous().to(device)
                rain_p = testr.to(device)
                labels = testy.to(device)
                if testm is not None:
                    mask = testm.to(device)
                else:
                    mask = None
            else:
                inputs = torch.as_tensor(testx.transpose(0, 3, 1, 2), dtype=torch.float).to(device)
                rain_p = torch.as_tensor(testr,  dtype=torch.float).to(device)
                labels = torch.as_tensor(testy,  dtype=torch.float).to(device)
                if testm is not None:
                    mask = torch.as_tensor(testm,  dtype=torch.float).to(device)
                else:
                    mask = None
                    
            outputs = model(inputs = inputs, rain_p = rain_p, labels = labels)
            loss  = outputs["loss"]
            preds = outputs["results"]
            tot_loss += loss
        else:
            preds = testx
        
        if testm is not None:
            notws = testm[:,:,:,0]==-1
            if torch.is_tensor(notws) == True:
                notws = notws.to(cpu_device).detach().numpy()
        else:
            notws = np.zeros((testy.shape[0], 256, 256))

        preds = torch.squeeze(preds, 1)
        preds = torch.unsqueeze(preds, 3)
        preds = preds.to(cpu_device).detach().numpy()
        if torch.is_tensor(testy) == True:
            testy = testy.to(cpu_device).detach().numpy()

        TOTAL=np.sum(np.logical_not(notws))
        l_TOTAL += TOTAL

        if len(testy.shape)==4:
            fl_obs=np.power(testy[:,:,:,0],2)>0.05
            fl_pred=np.power(preds[:,:,:,0],2)>0.05
        else:
            fl_obs=np.power(testy[:,:,:],2)>0.05
            fl_pred=np.power(preds[:,:,:],2)>0.05
        #compute error on flooded pixels only (RMSE and NSE)
        flselect=np.logical_or(fl_obs,fl_pred)
        flpix=flpix+np.sum(flselect)
        flSSE=flSSE+np.sum(np.power(np.power(testy[flselect,0],2)-np.power(preds[flselect,0],2),2))
        flSSE0=flSSE0+np.sum(np.power(np.power(testy[flselect,0],2),2))

        #
        H=np.logical_and(np.logical_and(fl_obs,np.logical_not(notws)),fl_pred)
        M=np.logical_and(np.logical_and(fl_obs,np.logical_not(notws)),np.logical_not(fl_pred))
        FP=np.logical_and(np.logical_and(np.logical_not(fl_obs),np.logical_not(notws)),fl_pred)
        TN=np.logical_and(np.logical_and(np.logical_not(fl_obs),np.logical_not(notws)),np.logical_not(fl_pred))
        NOBS=np.logical_and(np.logical_not(notws),fl_obs)
        NPRED=np.logical_and(np.logical_not(notws),fl_pred)
        NNOBS=np.logical_and(np.logical_not(fl_obs),np.logical_not(notws))
        #
        H=np.sum(H)
        M=np.sum(M)
        FP=np.sum(FP)
        TN=np.sum(TN)
        NOBS=np.sum(NOBS)
        NPRED=np.sum(NPRED)
        NNOBS=np.sum(NNOBS)
        #
        l_ACC_005 = l_ACC_005 + (H+TN)
        l_H_005   = l_H_005   + H
        l_M_005   = l_M_005   + M
        l_FP_005  = l_FP_005  + FP
        
        #
        if len(testy.shape)==4:
            fl_obs=np.power(testy[:,:,:,0],2)>0.30
            fl_pred=np.power(preds[:,:,:,0],2)>0.30
        else:
            fl_obs=np.power(testy[:,:,:],2)>0.30
            fl_pred=np.power(preds[:,:,:],2)>0.30
        H=np.logical_and(np.logical_and(fl_obs,np.logical_not(notws)),fl_pred)
        M=np.logical_and(np.logical_and(fl_obs,np.logical_not(notws)),np.logical_not(fl_pred))
        FP=np.logical_and(np.logical_and(np.logical_not(fl_obs),np.logical_not(notws)),fl_pred)
        TN=np.logical_and(np.logical_and(np.logical_not(fl_obs),np.logical_not(notws)),np.logical_not(fl_pred))
        NOBS=np.logical_and(np.logical_not(notws),fl_obs)
        NPRED=np.logical_and(np.logical_not(notws),fl_pred)
        NNOBS=np.logical_and(np.logical_not(fl_obs),np.logical_not(notws))
        #
        H=np.sum(H)
        M=np.sum(M)
        FP=np.sum(FP)
        TN=np.sum(TN)
        NOBS=np.sum(NOBS)
        NPRED=np.sum(NPRED)
        NNOBS=np.sum(NNOBS)
        #
        l_ACC_030 = l_ACC_030 + (H+TN)
        l_H_030   = l_H_030   + H
        l_M_030   = l_M_030   + M
        l_FP_030  = l_FP_030  + FP
        
        torch.cuda.empty_cache()
        
        return print_cnt, tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL
        
def eval_work(model, use_model_eval, train_generator, train_nsteps, val_generator, val_nsteps, results_path):
    #compute score values for evaluating model fit
    
    l_ACC_005 = 0; l_ACC_030 = 0
    l_H_005   = 0; l_H_030   = 0
    l_M_005   = 0; l_M_030   = 0
    l_FP_005  = 0; l_FP_030  = 0
    l_TOTAL   = 0
    flpix=0
    flSSE=0
    flSSE0 = 0
        
    istrain = False
    print_cnt = 0
    print("------------evaluate train")
    
    cnt = 0
    tot_loss = 0
    if use_model_eval is True:
        print("---model.eval")
        model.eval()
    else:
        print("---model.train")
        model.train()
        
    for batch in tqdm(train_generator):
        inputr = batch[1]
        
        print_cnt, tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = evaluate(
                model, batch[0], inputr, batch[2], None, istrain, print_cnt, tot_loss, 
                             flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, -1)
        cnt += 1
        if cnt == train_nsteps:
            break
    tot_loss = tot_loss / train_nsteps
    cal("train", tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, True)
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------")
    
    
    l_ACC_005 = 0; l_ACC_030 = 0
    l_H_005   = 0; l_H_030   = 0
    l_M_005   = 0; l_M_030   = 0
    l_FP_005  = 0; l_FP_030  = 0
    l_TOTAL   = 0
    flpix=0
    flSSE=0
    flSSE0 = 0
    
    istrain = False
    print_cnt = 0
    cnt = 0
    tot_loss = 0
    print("------------evaluate val")
    for batch in tqdm(val_generator):
        print_cnt, tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = evaluate(
                model, batch[0], batch[1], batch[2], batch[3], istrain, print_cnt, tot_loss, 
                             flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, -1)
        cnt += 1
        if cnt == val_nsteps:
            break
    
    tot_loss = tot_loss / val_nsteps
    cal("valid", tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, True)
    
    if results_path != "../results/eval_valid/None":
        results = {
            "RMSE": math.sqrt(div(flSSE, flpix)), 
            "ratio of flooding area": div(l_H_005+l_FP_005, l_H_005+l_M_005), 
            "AvgCSI": (div(l_H_005, (l_H_005+l_M_005+l_FP_005)) + div(l_H_030, (l_H_030+l_M_030+l_FP_030)))/2, 
            "CSI0.05": div(l_H_005, (l_H_005+l_M_005+l_FP_005)), 
            "CSI0.3": div(l_H_030, (l_H_030+l_M_030+l_FP_030))
            }
        with open(results_path, "w", encoding = "utf-8") as fout:
            json.dump(results, fout, indent = 4, ensure_ascii = False)

def eval_valid(model_test_path, model_class = MobileNetViTv2, temp_eval_batch = 10, use_model_eval = True, use_dataset = False, results_path = None):
    
    print(" results_path: %s"%(results_path))
    results_path = "../results/eval_valid/%s"%(results_path)
    
    if os.path.exists(results_path):
        with open(results_path, "r", encoding = "utf-8") as fout:
            results = json.load(fout)
        for i in results:
            print("    %s: %.3lf "%(i, results[i]))
        return 
    
    
    train_data_size = 100
    temp_batch = temp_eval_batch
    print(" eval batch: ", temp_batch)
    if use_dataset == True:
        train_dataset, train_steps, valid_dataset, valid_steps = make_dataset1(batch_size = temp_batch, eval_batch_size = temp_batch, train_data_size = train_data_size, valid_data_size = 29*5, augs = False, augr = 0, save_path = None)
        num_train_data     = len(train_dataset)
        num_valid_data     = len(valid_dataset)

    print(" num_train_data:%d"%(num_train_data))
    print(" num_valid_data:%d"%(num_valid_data))
    print(" train_steps:%d"%(train_steps))
    print(" valid_steps:%d"%(valid_steps))
    print(" model test path: %s"%(model_test_path))
    
    print(" model_class ", model_class)
    try:
        model = model_class()
        
        state_dict = torch.load(model_test_path)
        
        new_state_dict = {}
        ban_keys = ["attention", "classifier"]
        for key, value in state_dict.items():
            flag = 0
            for bk in ban_keys:
                if bk in key:
                    flag = 1
                    break
            if flag == 1:
                print(" not load %s"%(key))
                continue                
            new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
    except:
        model = torch.load("%s"%(model_test_path))
        
    device     = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cpu_device = torch.device("cpu")
    print("device:%s  cpu_device:%s"%(device, cpu_device))
    model.to(device)
    
    eval_work(model = model, use_model_eval = use_model_eval,
            train_generator = train_dataset, train_nsteps = train_steps, val_generator = valid_dataset, val_nsteps = valid_steps, results_path = results_path)

def eval_test14(model_test_path, model_class = MobileNetViTv2, temp_eval_batch = 10, use_model_eval = True, use_dataset = False, results_path = None): 
    train_data_size = 6664
    temp_batch = 1
    print(" results_path: %s"%(results_path))
    results_path = "../results/eval_test14/%s"%(results_path)
    
    print(" model test path: %s"%(model_test_path))
    print(" model_class ", model_class)
    print(" eval batch: ", temp_batch)
    if use_dataset == True:
        train_dataset, train_steps = make_dataset2(batch_size = temp_batch, eval_batch_size = temp_batch, train_data_size = train_data_size, valid_data_size = 29*5, save_path = "sp")
        num_train_data             = len(train_dataset)
        print(" train_steps:%d"%(train_steps))
    print(" num_train_data:%d"%(num_train_data))
    
    if not os.path.exists(results_path):
        try:
            model = model_class() 
            state_dict = torch.load(model_test_path)
            new_state_dict = {}
            ban_keys = ["attention", "classifier"]
            for key, value in state_dict.items():
                flag = 0
                for bk in ban_keys:
                    if bk in key:
                        flag = 1
                        break
                if flag == 1:
                    print(" not load %s"%(key))
                    continue                
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        except:
            print(" load model directly  %s"%(model_test_path))
            model = torch.load("%s"%(model_test_path))
            
            
        device     = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        cpu_device = torch.device("cpu")
        print("device:%s  cpu_device:%s"%(device, cpu_device))
        model.to(device)
    
    [select_train_rain,select_train_patch,select_val_rain,select_val_patch]=np.load("../data/selectors_regular.npy",allow_pickle=True)
    select_val_rain = np.unique(select_val_rain)
    select_val_patch = np.unique(select_val_patch)
    print(" select_val_rain ", len(select_val_rain), select_val_rain)
    print(" select_val_patch ", len(select_val_patch), select_val_patch)
    
    istrain = False
    print_cnt = 0
    
    cnt = 0
    if os.path.exists(results_path):
        with open(results_path, "r", encoding = "utf-8") as fout:
            results = json.load(fout)
    else:
        results = {
            "model_path": model_test_path,
            "train spatial terrain inputs---train rainfall inputs": -1,
            "valid spatial terrain inputs---train rainfall inputs": -1,
            "train spatial terrain inputs---valid rainfall inputs": -1,
            "valid spatial terrain inputs---valid rainfall inputs": -1,
            "metrics": []
        }
        if use_model_eval is True:
            print("---model.eval")
            model.eval()
        else:
            print("---model.train")
            model.train()
        for batch in tqdm(train_dataset):
            tot_loss = 0
            l_ACC_005 = 0; l_ACC_030 = 0
            l_H_005   = 0; l_H_030   = 0
            l_M_005   = 0; l_M_030   = 0
            l_FP_005  = 0; l_FP_030  = 0
            l_TOTAL   = 0
            flpix=0
            flSSE=0
            flSSE0 = 0
            print_cnt, tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = evaluate(
                    model, batch[0], batch[1], batch[2], batch[3], istrain, print_cnt, tot_loss, 
                                flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, -1)
            
            results["metrics"].append([float(tot_loss.cpu().numpy()), float(flSSE), float(flpix), float(flSSE0), int(l_ACC_005), int(l_ACC_030), int(l_H_005), int(l_H_030), int(l_M_005), int(l_M_030), int(l_FP_005), int(l_FP_030), int(l_TOTAL)])
            cnt += 1
            if cnt == train_steps:
                break
        
    print(" results for each patch ", select_val_patch)
    for i in range(119):
        tr = [0]*13
        for j in range(56):
            a = i+j*119
            for k in range(13):
                tr[k] += results["metrics"][a][k]
        tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = tr
        cal("patch-%d"%(i), tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, True)
    print("-------------------------------------------------------------------")
    print(" results for each rain ", select_val_rain)
    for j in range(56):
        tr = [0]*13
        for i in range(119):
            a = i+j*119
            for k in range(13):
                tr[k] += results["metrics"][a][k]
        tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = tr
        cal("rain-%d"%(j), tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, True)
    print("-------------------------------------------------------------------")
    print(" results for valid data")
    tr = [0]*13
    for j in select_val_rain:
        for i in select_val_patch:
            a = i+j*119
            for k in range(13):
                tr[k] += results["metrics"][a][k]
    tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = tr
    cal("valid_data", tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, True)
    print("-------------------------------------------------------------------")
    print(" results for different types of data")
    tr = []
    tr_cnt = [0]*4
    for i in range(4):
        tr.append([0]*13)
    for i in range(119):
        for j in range(3, 56):
            a = i+j*119
            r'''
            0 train spatial terrain inputs---train rainfall inputs
            1 valid spatial terrain inputs---train rainfall inputs
            2 train spatial terrain inputs---valid rainfall inputs
            3 valid spatial terrain inputs---valid rainfall inputs
            '''
            o = 0
            if j in select_val_rain:
                o += 2
            if i in select_val_patch:
                o += 1
            tr_cnt[o] += 1
            for k in range(13):
                tr[o][k] += results["metrics"][a][k]
    for i in range(4):
        ss = ""
        if i%2 == 0:
            ss += "train spatial terrain inputs"
        else:
            ss += "valid spatial terrain inputs"
        if i//2 == 0:
            ss += "---train rainfall inputs"
        else:
            ss += "---valid rainfall inputs"
        tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = tr[i]
        results[ss] = {
            "RMSE": math.sqrt(div(flSSE, flpix)), 
            "ratio of flooding area": div(l_H_005+l_FP_005, l_H_005+l_M_005), 
            "AvgCSI": (div(l_H_005, (l_H_005+l_M_005+l_FP_005)) + div(l_H_030, (l_H_030+l_M_030+l_FP_030)))/2, 
            "CSI0.05": div(l_H_005, (l_H_005+l_M_005+l_FP_005)), 
            "CSI0.3": div(l_H_030, (l_H_030+l_M_030+l_FP_030))
            }
        cal(ss, tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, True)
    
    if results_path != "../results/eval_test14//None":
        with open(results_path, "w", encoding = "utf-8") as fout:
            json.dump(results, fout, indent = 4, ensure_ascii = False)
    
def eval_test47(model_test_path, model_class = MobileNetViTv2, temp_eval_batch = 10, use_model_eval = True, use_dataset = False, results_path = None): 
    train_data_size = 100
    valid_data_size = 29*5
    temp_batch = 1
    print(" results_path: %s"%(results_path))
    results_path = "../results/eval_test47/%s"%(results_path)
    
    print(" model test path: %s"%(model_test_path))
    print(" model_class ", model_class)
    print(" eval batch: ", temp_batch)
    if use_dataset == True:
        train_dataset, train_steps, valid_dataset, valid_steps = make_dataset1(batch_size = 1, eval_batch_size = 1, train_data_size = train_data_size, valid_data_size = valid_data_size, augs = True, augr = 0, save_path = None, eval_aug2d = True)
        num_train_data     = len(train_dataset)
        num_valid_data     = len(valid_dataset)
        print(" train_steps:%d"%(train_steps))
        print(" valid_steps:%d"%(valid_steps))
    print(" num_train_data:%d"%(num_train_data))
    print(" num_valid_data:%d"%(num_valid_data))
    
    if not os.path.exists(results_path):
        try:
            model = model_class() 
            state_dict = torch.load(model_test_path)
            new_state_dict = {}
            ban_keys = ["attention", "classifier"]
            for key, value in state_dict.items():
                flag = 0
                for bk in ban_keys:
                    if bk in key:
                        flag = 1
                        break
                if flag == 1:
                    print(" not load %s"%(key))
                    continue                
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        except:
            print(" load model directly  %s"%(model_test_path))
            model = torch.load("%s"%(model_test_path))
            
            
        device     = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        cpu_device = torch.device("cpu")
        print("device:%s  cpu_device:%s"%(device, cpu_device))
        model.to(device)
    
    istrain = False
    print_cnt = 0
    
    cnt = 0
    if os.path.exists(results_path):
        with open(results_path, "r", encoding = "utf-8") as fout:
            results = json.load(fout)
    else:
        results = {
            "model_path": model_test_path,
            "train spatial terrain inputs---train rainfall inputs---original": -1,
            "train spatial terrain inputs---train rainfall inputs---synthetic": -1,
            "valid spatial terrain inputs---valid rainfall inputs---original": -1,
            "valid spatial terrain inputs---valid rainfall inputs---synthetic": -1,
            "metrics train": [],
            "metrics valid": [],
        }
        if use_model_eval is True:
            print("---model.eval")
            model.eval()
        else:
            print("---model.train")
            model.train()
        
        for dd in ["train", "valid"]:
            if dd == "train":
                dataset = train_dataset
                steps   = train_steps
            else:
                dataset = valid_dataset
                steps   = valid_steps
            cnt = 0
            
            for batch in tqdm(dataset):
                tot_loss = 0
                l_ACC_005 = 0; l_ACC_030 = 0
                l_H_005   = 0; l_H_030   = 0
                l_M_005   = 0; l_M_030   = 0
                l_FP_005  = 0; l_FP_030  = 0
                l_TOTAL   = 0
                flpix=0
                flSSE=0
                flSSE0 = 0
                
                inputs = batch[0].permute(0, 3, 1, 2).contiguous()
                if len(batch[1].shape) == 3:
                    rain_p = batch[1][:, 0, :].to(device, non_blocking=True)
                else:
                    rain_p = batch[1].to(device, non_blocking=True)
                labels = batch[2].to(device, non_blocking=True)
                inputs = inputs.to(device, non_blocking=True)
                with torch.no_grad():
                    output = model(inputs = inputs, rain_p = rain_p, labels = labels)
                tot_loss += output["loss"]
                print_cnt, tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = evaluate(
                                    None, output["results"], None, batch[2], batch[3], istrain, print_cnt, tot_loss, 
                                    flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, -1)
                results["metrics %s"%(dd)].append([float(tot_loss.cpu().numpy()), float(flSSE), float(flpix), float(flSSE0), int(l_ACC_005), int(l_ACC_030), int(l_H_005), int(l_H_030), int(l_M_005), int(l_M_030), int(l_FP_005), int(l_FP_030), int(l_TOTAL)])
                
                cnt += 1
                if cnt == steps:
                    break
        
    
    print(" results for different types of data")
    tr = []
    tr_cnt = [0]*4
    for i in range(4):
        tr.append([0]*13)
    for dd in ["train", "valid"]:
        if dd == "train":
            i = 0
        else:
            i = 2
            
        for a in range(len(results["metrics %s"%(dd)])):
            o = i
            if a%8 > 0:
                o += 1
            tr_cnt[o] += 1
            for k in range(13):
                tr[o][k] += results["metrics %s"%(dd)][a][k]
                
    for i in range(4):
        ss = ""
        
        if i//2 == 0:
            ss += "train spatial terrain inputs---train rainfall inputs"
        else:
            ss += "valid spatial terrain inputs---valid rainfall inputs"
            
        if i%2 == 0:
            ss += "---original"
        else:
            ss += "---synthetic"
            
        tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL = tr[i]
        results[ss] = {
            "RMSE": math.sqrt(div(flSSE, flpix)), 
            "ratio of flooding area": div(l_H_005+l_FP_005, l_H_005+l_M_005), 
            "AvgCSI": (div(l_H_005, (l_H_005+l_M_005+l_FP_005)) + div(l_H_030, (l_H_030+l_M_030+l_FP_030)))/2, 
            "CSI0.05": div(l_H_005, (l_H_005+l_M_005+l_FP_005)), 
            "CSI0.3": div(l_H_030, (l_H_030+l_M_030+l_FP_030))
            }
        cal(ss, tot_loss, flSSE, flpix, flSSE0, l_ACC_005, l_ACC_030, l_H_005, l_H_030, l_M_005, l_M_030, l_FP_005, l_FP_030, l_TOTAL, True)
    
    if results_path != "../results/eval_test47/None":
        with open(results_path, "w", encoding = "utf-8") as fout:
            json.dump(results, fout, indent = 4, ensure_ascii = False)
            
if __name__=="__main__":
    
    # choose model structure (MobileNetViTv2 or ResU-Net) and parameters
    r'''
    model_class = ResUNet
    model_test_path = "../models/ResU-Net-augs.bin"
    '''
    
    model_class = MobileNetViTv2
    model_test_path = "../models/MobileViTv2-basic.bin"
    #model_test_path = "../models/MobileViTv2-augs.bin"
    
    eval_valid(model_test_path, model_class, 5, True, True, results_path = model_test_path.replace("../models/", "").replace(".bin", ".json"))# experiments in Section 3.2 and Appendix A to assess model performance on validation dataset
    #eval_test14(model_test_path, model_class, 5, True, True, results_path = model_test_path.replace("../models/", "").replace(".bin", ".json")) # experiments in Section 3.3 to assess model performance on test dataset 1-4
    #eval_test47(model_test_path, model_class, 5, True, True, results_path = model_test_path.replace("../models/", "").replace(".bin", ".json")) # experiments in Section 3.3 to assess model performance on test dataset 4-7