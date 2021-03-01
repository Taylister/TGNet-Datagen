import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms
from utils import *
import cfg
from tqdm import tqdm

from models import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from torchvision import models, transforms, datasets
import torchvision.transforms.functional as F
from dataset import CSNet_dataset, Example_dataset, To_tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def main(args):
    
    print('===> Loading datasets')
 
    trfms = To_tensor()
    example_data = Example_dataset(data_dir = args.input_dirpath, transform = trfms)    
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)


    print('===> Loading models')

    if cfg.gpu :
        gpu = torch.device("cuda:0")
        net_g = define_G(3, 1, 64,'batch', False, 'normal', 0.02, gpu_id=gpu)
        net_g.load_state_dict(torch.load(args.model))
        #net_g.load_state_dict(torch.load(cfg.predict_ckpt_path))
       
    else:
        net_g = define_G(3, 1, 64,'batch', False, 'normal', 0.02, gpu_id='cpu')
        net_g.load_state_dict(torch.load(args.model, map_location='cpu'))
        #net_g.load_state_dict(torch.load(cfg.predict_ckpt_path, map_location='cpu'))
        

    savedir = args.output_dirpath
         
    example_iter = iter(example_loader)
    net_g.eval()
    torch.set_grad_enabled(False)

    pbar = tqdm(len(example_iter),total=len(example_iter))
    pbar.set_description("Start segmentation of character region")
    for ex_iter, batch in enumerate(example_iter):
        
        if cfg.gpu:
            i_s = batch[0].to(gpu)
            name = str(batch[1][0])
     
            o_mask = net_g(i_s)
            o_mask = o_mask.squeeze(0).to('cpu')
        else:
            i_s = batch[0]
            name = str(batch[1][0])
     
            o_mask = net_g(i_s)
            o_mask = o_mask.squeeze(0)
        
                 
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        o_mask = F.to_pil_image(o_mask)
        o_mask.save(os.path.join(savedir, name + '.jpg'))
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    
    parser = ArgumentParser()
    parser.add_argument(
        'input_dirpath',
        type=str,
        help='path to image_folder which contains text images'
    )
    parser.add_argument(
        'output_dirpath',
        type=str,
        help='path to output folder'
    )
    parser.add_argument(
        'model',
        type=str,
        help='path to pretraind model'
    )
    args = parser.parse_args()
    #python3 predict.py "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/analysis/proposed/network_output" "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/analysis/edge_detection" "weight/CSNet_weight.pth"
    #python3 predict.py "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/dataset/extracted_title/test" "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/analysis/proposed/CSNet_out_t" "weight/CSNet_weight.pth"
    # python3 predict.py "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/analysis/4(no_skeleton)/network_output" "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/analysis/4(no_skeleton)/CSNet_out" "weight/CSNet_weight.pth"
    #python3 predict.py "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/analysis/training/train_sample/o_t" "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/analysis/training/train_sample/CSNet_g" "weight/CSNet_weight.pth"
    #python3 predict.py "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/analysis/training/train_sample/o_t" "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/analysis/training/train_sample/CSNet_t" "weight/CSNet_weight.pth"
    #python3 predict.py "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/dataset/extracted_title/train" "/home/miyazonotaiga/デスクトップ/MyResearch/TGNet/analysis/training/CSNet_output" "weight/CSNet_weight.pth"

    # python3 predict.py "../../TGNet-analysis/titles/baseline" "../../TGNet-analysis/masks/baseline" "weight/CSNet_weight.pth"

    # python3 predict.py "../../TGNet-analysis/titles/no_style" "../../TGNet-analysis/masks/no_style" "weight/CSNet_weight.pth"

    # python3 predict.py "../../TGNet-analysis/titles/no_discriminator" "../../TGNet-analysis/masks/no_discriminator" "weight/CSNet_weight.pth"

    # python3 predict.py "../../TGNet-analysis/titles/no_VGG" "../../TGNet-analysis/masks/no_VGG" "weight/CSNet_weight.pth"
    
    # python3 predict.py "../../TGNet-analysis/titles/no_skeleton" "../../TGNet-analysis/masks/no_skeleton" "weight/CSNet_weight.pth"
    
    main(args)