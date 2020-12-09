
import numpy as np
import os
import torch
import torchvision.transforms
import cfg
from tqdm import tqdm
from datetime import datetime
from skimage.transform import resize
from skimage import io

def custom_collate(batch):
    i_s_batch, mask_t_batch = [], []
    
    w_sum = 0

    for item in batch:

        i_s= item[0]
        h, w = i_s.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        w_sum += int(w * scale_ratio)
        
    to_h = cfg.data_shape[0]
    to_w = w_sum // cfg.batch_size
    to_w = int(round(to_w / 8)) * 8
    to_scale = (to_h, to_w)
    
    for item in batch:
   
        i_s, mask_t = item

        i_s = resize(i_s, to_scale, preserve_range=True)
        mask_t = np.expand_dims(resize(mask_t, to_scale, preserve_range=True), axis = -1)

        i_s = i_s.transpose((2, 0, 1))
        mask_t = mask_t.transpose((2, 0, 1)) 

        i_s_batch.append(i_s) 
        mask_t_batch.append(mask_t)

    i_s_batch = np.stack(i_s_batch)
    mask_t_batch = np.stack(mask_t_batch)

    i_s_batch = torch.from_numpy(i_s_batch.astype(np.float32) / 127.5 - 1.) 
    mask_t_batch =torch.from_numpy(mask_t_batch.astype(np.float32) / 255.)    

    return [i_s_batch, mask_t_batch]

def get_train_name():
    return datetime.now().strftime('%Y%m%d%H%M%S')