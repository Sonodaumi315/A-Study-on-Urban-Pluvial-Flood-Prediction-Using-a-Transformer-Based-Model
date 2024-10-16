import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
import torch

class UfloodDataset(Dataset):
    
    def __init__(self, fx, fy, fm, no_inputs, l_rain_data, rowno_patchfiles, indices_r, rnvar, tsize, mask=False, augs=False, augr=0):
        
        self.len = len(rowno_patchfiles)
        if augs == True:
            self.len = self.len*8
        
        self.fx = fx
        self.fy = fy
        self.fm = fm
        self.no_inputs = no_inputs
        self.l_rain_data = l_rain_data
        
        self.rowno_patchfiles = rowno_patchfiles
        self.indices_r = indices_r
        self.rnvar = rnvar
        self.tsize = tsize
        
        self.mask = mask
        self.augs  = augs 
        self.augr = augr
        
        xf2=np.memmap(fx, mode="readonly",dtype=np.float32, shape=(np.unique(rowno_patchfiles).shape[0],tsize,tsize,no_inputs))
        self.xf=np.copy(xf2)
        del xf2
        yf2=np.memmap(fy, mode="readonly",dtype=np.float32, shape=(rowno_patchfiles.shape[0],tsize,tsize))
        self.yf=np.copy(yf2)
        del yf2
        if mask: 
            mf2=np.memmap(fm, mode="readonly",dtype=np.float32, shape=(rowno_patchfiles.shape[0],tsize,tsize))
            self.mf=np.copy(mf2)
            del mf2
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.augs == True:
            rot   = (index%8)%4
            flp   = (index%8)//4
            index = index // 8
            
        patch_sel=self.rowno_patchfiles[index,0]    
        x1=self.xf[patch_sel,:,:,:].copy()
        y=self.yf[index,:,:].copy()
        y=np.expand_dims(y,axis=2)
        
        r_sel=self.indices_r[index]      
        if self.augr == 0:
            r=np.zeros(shape=(self.rnvar))
            rvec=self.l_rain_data[r_sel]
            r[:]=np.array(rvec)
        else:
            r=np.zeros(shape=(self.augr*2+1, self.rnvar))
            rvec=self.l_rain_data[r_sel]
            r[:, :]=np.array(rvec)
        
        x1 = x1.astype("float32")
        r  = r.astype("float32")
        y  = y.astype("float32")
        if self.mask:
            m=self.mf[index,:,:].copy()
            m=np.expand_dims(m,axis=2)
            m = m.astype("float32")
            
        if self.augs == True:
            x1 = np.rot90(x1, rot, [0, 1])
            y  = np.rot90(y,  rot, [0, 1])
            if self.mask:
                m = np.rot90(m, rot, [0, 1])
            
            if flp == 1:
                x1 = np.flip(x1, 0)
                y  = np.flip(y,  0)
                if self.mask:
                    m = np.flip(m, 0)
            
            x1 = np.ascontiguousarray(x1)
            y  = np.ascontiguousarray(y)
            if self.mask:
                m = np.ascontiguousarray(m)
            if index < 0:
                pl = 3
                for i in list(range(pl)) + list(range(256-pl, 256)):
                    ss = ""
                    for j in list(range(pl)) + list(range(256-pl, 256)):
                        ss += "%.3lf "%(x1[i][j][0])
                for i in list(range(pl)) + list(range(256-pl, 256)):
                    ss = ""
                    for j in list(range(pl)) + list(range(256-pl, 256)):
                        ss += "%.3lf "%(y[i][j][0])                
            
        if self.mask:
            return (x1,r,y,m)
        else:                    
            return (x1,r,y)
        
def make_dataset1(batch_size = 16, eval_batch_size = 10, train_data_size = None, valid_data_size = None, dist = False, nprocs = 1, augs = False, augr = 0, save_path = None, delta_p = 0.01, eval_aug2d = False):
    #seed = 233
    num_workers = 1
    #np.random.seed(seed)
    #print(" seed:%d"%(seed))
    print(" num_workers:%d "%(num_workers))
    print(" augr:%d"%(augr))
    print(" delta_p:%.3lf"%(delta_p))
    
    workfolder=os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(workfolder,'libs'))
    sys.path.append(workfolder)
    inputnames=['curv','imp','slope_flacc_w','sinkdepth','cos','sin']
    fend=2
    ##################################################################
    import libs.rainvec as rainvec
    import libs.rainvec_nat as rainvec_nat
    import libs.manage_data as md, libs.sample_training_data as samp
    ##################################################################
    tsize=256 #target edge length for image
    random_patches=True

    npzfile = np.load(os.path.join(workfolder, os.pardir, "data", 'Datafiles', 'Flood', 'events.npz'))
    evnames=npzfile['evnames']#;nevent=evnames.shape[0]
    del npzfile

    #rain data - results for CDS storms are not used in training and validation
    rainlist=[]
    for ii in range(evnames.shape[0]):
        if 'CDS' in evnames[ii]:
            rain=rainvec.rainvec(int(evnames[ii][7:10]),'CDS_M') 
        elif 'NAT' in evnames[ii]:
            rain=rainvec_nat.rainvec_nat(int(evnames[ii][4:6]))
        else:
            sys.exit('wrong event type')
        rainlist.append([x/25.0 for x in rain]) #biggest rainfall is 25mm/10min
    rlength=max([len(x) for x in rainlist])
    rainvariables=[md.make_rain_variables2(x) for x in rainlist]
    
    rvar_scale=[max([xx[x] for xx in rainvariables])-min([xx[x] for xx in rainvariables]) for x in range(9)]
    rainvariables=[[x[xx]/rvar_scale[xx] for xx in range(9)] for x in rainvariables]
    
    origin_rainvariables = rainvariables
    
    rainlist = [[x] for x in rainlist]
    rainvariables = [[x] for x in rainvariables]
    
    def get_increase_rain(origin_rainlist, update_ratio):
        rl = len(origin_rainlist)
        weight = []
        new_rainlist = []
        for i in range(rl):
            new_rainlist.append(origin_rainlist[i])
            weight.append(abs(np.random.randn()))
        totp = sum(origin_rainlist)
        totw = sum(weight)
        for i in range(rl):
            new_rainlist[i] += totp*update_ratio*weight[i]/totw
        return new_rainlist
        
    def get_decrease_rain(origin_rainlist, update_ratio):
        rl = len(origin_rainlist)
        weight = []
        new_rainlist = []
        for i in range(rl):
            new_rainlist.append(origin_rainlist[i])
            if new_rainlist[-1] == 0:
                weight.append(0)
            else:
                weight.append(abs(np.random.randn()))
        totp = sum(origin_rainlist)

        totw = sum(weight)
        for i in range(rl):
            new_rainlist[i] = max(0, new_rainlist[i]-totp*update_ratio*weight[i]/totw)
        return new_rainlist
    
    for i in range(len(rainlist)):
        #increase
        for j in range(augr):
            update_ratio = delta_p*(j+1)/augr
            tempr = get_increase_rain(origin_rainlist = rainlist[i][0], update_ratio = update_ratio)
            tempvar = md.make_rain_variables2(tempr)
            tempvar = [tempvar[xx]/rvar_scale[xx] for xx in range(9)]
            rainlist[i].append(tempr)
            rainvariables[i].append(tempvar)
        #decrease
        for j in range(augr):
            update_ratio = delta_p*(j+1)/augr
            tempr = get_decrease_rain(origin_rainlist = rainlist[i][0], update_ratio = update_ratio)
            tempvar = md.make_rain_variables2(tempr)
            tempvar = [tempvar[xx]/rvar_scale[xx] for xx in range(9)]
            rainlist[i].append(tempr)
            rainvariables[i].append(tempvar)
            
    rnvar=9

    ##########################################################################
    #get selectors where patches are defined according to grid layout and randomly select some of the grid patches for validation
    import libs.sample_training_data as samp
    name_selectors_regular = 'data/selectors_regular.npy'
    if os.path.exists(os.path.join(workfolder,os.pardir,name_selectors_regular)):
        [select_train_rain,select_train_patch,select_val_rain,select_val_patch]=np.load(os.path.join(workfolder,os.pardir,name_selectors_regular),allow_pickle=True)
    else:
        print(" not exist %s"%(name_selectors_regular))
        return None, 0, None, 0
    
    ########################
    #create patch-extents and selectors for training with random and overlapping patches, and random combinations of patches and rain
    #this step takes one hour to execute, but needs to be performed only once
    npatchtrain=train_data_size#10000   
    name_selectors_patchlist = 'data/selectors_patchlist.npy'
      
    if random_patches:
        valpatches=np.unique(select_val_patch) #indices in patchlist that will be used for validation
        if os.path.exists(os.path.join(workfolder,os.pardir,name_selectors_patchlist)):
            print("  exists----%s"%(os.path.join(workfolder,os.pardir,name_selectors_patchlist)))
            [select_train_rain,select_train_patch,select_val_rain,select_val_patch,patchlist2]=np.load(os.path.join(workfolder,os.pardir,name_selectors_patchlist),allow_pickle=True)
        else:
            #put the validation patches in the beginning of the new patch list, add 200 randomly placed training patches
            print("  not exists----%s"%(os.path.join(workfolder,os.pardir,name_selectors_patchlist)))
            return None, 0, None, 0
    
    select_train_rain  = select_train_rain[:npatchtrain]
    select_train_patch = select_train_patch[:npatchtrain]

    fend = str(fend)
    patchfx     = "px"+fend+".arr"
    patchfy     = "py"+fend+".arr"
    patchfm     = "pm"+fend+".arr"
    patchfx_val = "pxval"+fend+".arr"
    patchfy_val = "pyval"+fend+".arr"
    patchfm_val = "pmval"+fend+".arr"

    if save_path is not None:
        patchfx     = "%s_%s"%(save_path, patchfx)
        patchfy     = "%s_%s"%(save_path, patchfy)
        patchfm     = "%s_%s"%(save_path, patchfm)
        patchfx_val = "%s_%s"%(save_path, patchfx_val)
        patchfy_val = "%s_%s"%(save_path, patchfy_val)
        patchfm_val = "%s_%s"%(save_path, patchfm_val)
    
    if len(select_train_rain) != 10000:
        patchfx = patchfx.replace(".arr", "-%d.arr"%(train_data_size))
        patchfy = patchfy.replace(".arr", "-%d.arr"%(train_data_size))
        patchfm = patchfm.replace(".arr", "-%d.arr"%(train_data_size))
        patchfx_val = patchfx_val.replace(".arr", "-%d.arr"%(train_data_size))
        patchfy_val = patchfy_val.replace(".arr", "-%d.arr"%(train_data_size))
        patchfm_val = patchfm_val.replace(".arr", "-%d.arr"%(train_data_size))
    patchfx     =os.path.join(workfolder, os.pardir, "data", patchfx)
    patchfy     =os.path.join(workfolder, os.pardir, "data", patchfy)
    patchfm     =os.path.join(workfolder, os.pardir, "data", patchfm)
    patchfx_val =os.path.join(workfolder, os.pardir, "data", patchfx_val)
    patchfy_val =os.path.join(workfolder, os.pardir, "data", patchfy_val)
    patchfm_val =os.path.join(workfolder, os.pardir, "data", patchfm_val)
    
    print(" len(select_train_patch) ", len(select_train_patch), len(select_train_rain))
    print(" len(select_val_patch)   ", len(select_val_patch),   len(select_val_rain))

    if os.path.exists(patchfx):
        print("  exists----patchfx:%s"%(patchfx))
        select_train_patchrows,tmp=md.get_patch_indices(select_train_patch)
        select_val_patchrows,tmp=md.get_patch_indices(select_val_patch)
    else:    
        print("  not exists----patchfx:%s"%(patchfx))
        
        npzfile = np.load(os.path.join(workfolder, os.pardir, "data", 'X_data.npz'))
        X_dict={}
        for entry in npzfile.files: X_dict[entry]=npzfile[entry]
        npzfile = np.load(os.path.join(workfolder, os.pardir, "data", 'wsheds.npz'))
        wshedarr=npzfile['wshedarr']
        del npzfile 

        Y_data=np.memmap(os.path.join(workfolder, os.pardir, "data", 'Datafiles', 'Flood', "Y_data.arr"), mode="readonly",dtype=np.float32, shape=(evnames.shape[0],X_dict['dem'].shape[0], X_dict['dem'].shape[1]))
        
        wsexclude=[-999,7,219] #wshed ID's that should not be considered - catchment 7 is a lake, 219 is a park with several sinks next to Lindved Å
        
        wshedarr2=wshedarr.copy()
        for ext in patchlist2[0:(valpatches.shape[0]-1)]: wshedarr2[ext[0]:ext[1],ext[2]:ext[3]]=-999
    
        print("     make train patch")
        select_train_patchrows=md.make_patch_files(patchfx,patchfy,patchfm,X_dict,inputnames,Y_data,select_train_patch,select_train_rain,patchlist2,wsexclude,wshedarr2,tsize,sieve=True)
        print("     make valid patch")
        select_val_patchrows=md.make_patch_files(patchfx_val,patchfy_val,patchfm_val,X_dict,inputnames,Y_data,select_val_patch,select_val_rain,patchlist2,wsexclude,wshedarr,tsize,sieve=True)
    
    
    if train_data_size is None:
        train_dataloader = None
        train_steps = 0
    else:
        st_time = time.time()
        dataset_cache = "uflood_train_data_%d"%(train_data_size)
        if augs == True:
            dataset_cache = "%s_augs"%(dataset_cache)
        if augr > 0:
            dataset_cache = "%s_augr%d"%(dataset_cache, augr)
        if save_path is not None:
            dataset_cache = "%s_%s"%(save_path, dataset_cache)
        dataset_cache = os.path.join(workfolder, os.pardir, "data", dataset_cache)
        
        if os.path.isfile(dataset_cache):
            print(" load dataset_cache:%s"%(dataset_cache))
            train_dataset = torch.load(dataset_cache)
        else:
            print(" dataset_cache:%s not exists"%(dataset_cache))
            train_dataset = UfloodDataset(patchfx,patchfy,patchfm,len(inputnames),rainvariables,select_train_patchrows,select_train_rain,rnvar,tsize,eval_aug2d,augs, augr)
            torch.save(train_dataset, dataset_cache, pickle_protocol = 4)
            
        st_time2 = time.time()
        print("   time for make Uflood train dataset ", st_time2-st_time)
        if augs == True:
            train_steps = len(select_train_patchrows)*8
        else:
            train_steps = len(select_train_patchrows)
            
        if dist == True:
            train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
            train_steps = train_steps // (batch_size*nprocs)
        else:
            if eval_aug2d == True:
                train_dataloader = DataLoader(train_dataset, shuffle=False,  batch_size=batch_size, drop_last=True)
            else:
                train_dataloader = DataLoader(train_dataset, shuffle=True,  batch_size=batch_size, drop_last=True)
            train_steps = train_steps // batch_size
        
        print("   time for make train dataloader ", time.time()-st_time2)
        
    if valid_data_size is None:
        valid_dataloader = None
        valid_steps = 0
    else:
        st_time = time.time()
        valid_dataset = UfloodDataset(patchfx_val,patchfy_val,patchfm_val,len(inputnames),origin_rainvariables,select_val_patchrows,select_val_rain,rnvar,tsize,mask=True,augs=eval_aug2d)
        st_time2 = time.time()
        print("   time for make Uflood valid dataset ", st_time2-st_time)
        if augs == True:
            valid_steps = len(select_val_patchrows)*8
        else:
            valid_steps = len(select_val_patchrows)
        if dist == True:
            valid_sampler = DistributedSampler(valid_dataset)
            valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=eval_batch_size, drop_last=False, num_workers=num_workers, pin_memory=True, sampler=valid_sampler)
            valid_steps = valid_steps // (eval_batch_size*nprocs)
        else:
            valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=eval_batch_size, drop_last=False)
            valid_steps = valid_steps // eval_batch_size
        print("   time for make valid dataloader ", time.time()-st_time2)
    print(" train_dataset  len:%d=batch_size:%d x steps:%d"%(batch_size*train_steps,      batch_size,      train_steps))
    print(" valid_dataset  len:%d=batch_size:%d x steps:%d"%(eval_batch_size*valid_steps, eval_batch_size, valid_steps))
    #return train_dataset, train_steps, val_dataset, val_steps
    if dist == True:
        return train_dataloader, train_steps, train_sampler, valid_dataloader, valid_steps, valid_sampler
    else:
        return train_dataloader, train_steps, valid_dataloader, valid_steps

def make_dataset2(batch_size = 1, eval_batch_size = 5, train_data_size = 119*56, valid_data_size = 119*56, dist = False, nprocs = 1, augs = False, save_path = None):
    workfolder=os.path.dirname(os.path.realpath(__file__))
    #####
    sys.path.append(os.path.join(workfolder,'libs'))
    sys.path.append(workfolder)
    ##################################################################
    #tf.config.list_physical_devices('GPU')
    ##################################################################
    #topographic inputs that should be used in the model
    inputnames=['curv','imp','slope_flacc_w','sinkdepth','cos','sin'] 
    fend=2
    ##################################################################
    import libs.rainvec as rainvec
    import libs.rainvec_nat as rainvec_nat
    import libs.manage_data as md, libs.sample_training_data as samp
    ##################################################################
    tsize=256 #target edge length for image
    random_patches=False
    wsexclude=[-999,7,219] #wshed ID's that should not be considered - catchment 7 is a lake, 219 is a park with several sinks next to Lindved Å
    ##################################################################
    
    npzfile = np.load(os.path.join(workfolder, os.pardir, "data", 'X_data.npz'))
    X_dict={}
    for entry in npzfile.files: X_dict[entry]=npzfile[entry]
    #del npzfile 
    npzfile = np.load(os.path.join(workfolder, os.pardir, "data",'wsheds.npz'))
    wshedarr=npzfile['wshedarr']
    del npzfile 
    npzfile = np.load(os.path.join(workfolder, os.pardir, "data", 'Datafiles','Flood','events.npz'))
    evnames=npzfile['evnames']#;nevent=evnames.shape[0]
    del npzfile
    Y_data = np.memmap(os.path.join(workfolder, os.pardir, "data", 'Datafiles','Flood',"Y_data.arr"), mode="readonly",dtype=np.float32, shape=(evnames.shape[0],X_dict['dem'].shape[0], X_dict['dem'].shape[1]))
    ##################################################################
    #get lists of watershed extent, ID and areas of each watershed
    indlist,wshedID,area=md.get_wshed_properties(wshedarr,wsexclude,1e6)
    patchlist=md.get_patch_list(wshedarr,wsexclude,tsize) #list of valid patchextents for training
    print(" patchlist ", len(patchlist), patchlist)
    ##################################################################
    #rain data - results for CDS storms are not used in training and validation
    rainlist=[]
    for ii in range(evnames.shape[0]):
        if 'CDS' in evnames[ii]:
            rain=rainvec.rainvec(int(evnames[ii][7:10]),'CDS_M') 
        elif 'NAT' in evnames[ii]:
            rain=rainvec_nat.rainvec_nat(int(evnames[ii][4:6]))
        else:
            sys.exit('wrong event type')
        rainlist.append([x/25.0 for x in rain]) #biggest rainfall is 25mm/10min
    rlength=max([len(x) for x in rainlist])
    rainvariables=[md.make_rain_variables2(x) for x in rainlist]
    rvar_scale=[max([xx[x] for xx in rainvariables])-min([xx[x] for xx in rainvariables]) for x in range(9)]
    rainvariables=[[x[xx]/rvar_scale[xx] for xx in range(9)] for x in rainvariables]
    rnvar=9

    ##########################################################################
    #get selectors where patches are defined according to grid layout and randomly select some of the grid patches for validation
    name_selectors_regular = "data/sp_selectors_regular-%d.npy"%(train_data_size)
    print(" name_selectors_regular ", name_selectors_regular)
    if os.path.exists(os.path.join(workfolder,os.pardir,name_selectors_regular)):
        [select_train_rain,select_train_patch]=np.load(os.path.join(workfolder,os.pardir,name_selectors_regular),allow_pickle=True)
    else:
        select_train_patch = np.tile(  np.arange(len(patchlist)), len(rainlist))
        select_train_rain  = np.repeat(np.arange(len(rainlist)),  len(patchlist))
        save_data = np.asarray([select_train_rain,select_train_patch], dtype = object)
        np.save(os.path.join(workfolder,os.pardir,name_selectors_regular), save_data)
    patchlist2 = patchlist
    
    ########################
    #create patch-extents and selectors for training with random and overlapping patches, and random combinations of patches and rain
    #this step takes one hour to execute, but needs to be performed only once          
    print(" random_patches ", random_patches,   len(select_train_patch), select_train_patch)
    print(" select_train_rain ",  len(select_train_rain),  select_train_rain[:10],  select_train_rain[-10:],  min(select_train_rain),  max(select_train_rain))
    print(" select_train_patch ", len(select_train_patch), select_train_patch[:10], select_train_patch[-10:], min(select_train_patch), max(select_train_patch))
    print(" patchlist2 ", len(patchlist2))
    print("             valid ", patchlist2[:3], patchlist2[26:29])
    print("             train ", patchlist2[29:34])

    fend = str(fend)
    
    patchfx     = "px"+fend+".arr"
    patchfy     = "py"+fend+".arr"
    patchfm     = "pm"+fend+".arr"
    patchfx = patchfx.replace(".arr", "-%d.arr"%(train_data_size))
    patchfy = patchfy.replace(".arr", "-%d.arr"%(train_data_size))
    patchfm = patchfm.replace(".arr", "-%d.arr"%(train_data_size))
    
    if save_path is not None:
        patchfx     = "%s_%s"%(save_path, patchfx)
        patchfy     = "%s_%s"%(save_path, patchfy)
        patchfm     = "%s_%s"%(save_path, patchfm)
    patchfx = os.path.join(workfolder, os.pardir, "data", patchfx)
    patchfy = os.path.join(workfolder, os.pardir, "data", patchfy)
    patchfm = os.path.join(workfolder, os.pardir, "data", patchfm)
    
    print(" len(select_train_patch) ", len(select_train_patch), len(select_train_rain), train_data_size)

    if os.path.exists(patchfx):
        print("  exists----patchfx:%s"%(patchfx))
        select_train_patchrows,tmp=md.get_patch_indices(select_train_patch)
    else:    
        print("  not exists----patchfx:%s"%(patchfx))
        select_train_patchrows=md.make_patch_files(patchfx,patchfy,patchfm,X_dict,inputnames,Y_data,select_train_patch,select_train_rain,patchlist2,wsexclude,wshedarr,tsize,sieve=True)
    
    print(" len(select_train_patchrows)  %d"%(len(select_train_patchrows)))
    print(" len(select_train_rain)  %d"%(len(select_train_rain)))
    print(" select_train_patchrows ", len(select_train_patchrows), select_train_patchrows[:5], select_train_patchrows[-5:])
    
    st_time = time.time()
    if augs == True:
        dataset_cache = "uflood_train_data_%d_aug8"%(train_data_size)
    else:
        dataset_cache = "uflood_train_data_%d"%(train_data_size)
    if save_path is not None:
        dataset_cache = "%s_%s"%(save_path, dataset_cache)
    dataset_cache = os.path.join(workfolder, os.pardir, "data", dataset_cache)
        
    if os.path.isfile(dataset_cache):
        print(" load dataset_cache:%s"%(dataset_cache))
        train_dataset = torch.load(dataset_cache)
    else:
        print(" dataset_cache:%s not exists"%(dataset_cache))
        train_dataset = UfloodDataset(patchfx,patchfy,patchfm,len(inputnames),rainvariables,select_train_patchrows,select_train_rain,rnvar,tsize,True,augs)
        torch.save(train_dataset, dataset_cache, pickle_protocol = 4)
    st_time2 = time.time()
    print("   time for make Uflood train dataset ", st_time2-st_time)
    if augs == True:
        train_steps = len(select_train_patchrows)*8
    else:
        train_steps = len(select_train_patchrows)
        
    if dist == True:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=4, pin_memory=True, sampler=train_sampler)
        train_steps = train_steps // (batch_size*nprocs)
    else:
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
        train_steps = train_steps // batch_size
    print("   time for make train dataloader ", time.time()-st_time2)
    
    print(" train_dataset  len:%d=batch_size:%d x steps:%d"%(batch_size*train_steps,      batch_size,      train_steps))
    
    return train_dataloader, train_steps