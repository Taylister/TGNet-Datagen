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


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        

def clip_grad(model):
    
    for h in model.parameters():
        h.data.clamp_(-0.01, 0.01)

def main():
    # ================================================
    # Preparation
    # ================================================
    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available.')
    
    gpu = torch.device('cuda:0')

    train_name = get_train_name()
 
    print('===> Loading datasets')
    train_data = CSNet_dataset(cfg, torp='train')
    train_data = DataLoader(dataset = train_data, batch_size = cfg.batch_size, shuffle = False, collate_fn = custom_collate,  pin_memory = True)
    
    trfms = To_tensor()
    example_data = Example_dataset(transform = trfms)    
    example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)


    print('===> Building models')

    if cfg.train_ckpt_G_path is None:
        net_g = define_G(3, 1, 64,'batch', False, 'normal', 0.02, gpu_id=gpu)
    else:
        net_g = torch.load(cfg.train_ckpt_G_path).to(gpu)
    
    if cfg.train_ckpt_D_path is None:
        net_d = define_D(3 + 1, 64, 'basic', gpu_id=gpu)
    else:
        net_d = torch.load(cfg.train_ckpt_D_path).to(gpu)
        
    criterionGAN = GANLoss().to(gpu)
    criterionL1 = nn.L1Loss().to(gpu)
    criterionMSE = nn.MSELoss().to(gpu)

    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, cfg)
    net_d_scheduler = get_scheduler(optimizer_d, cfg)

    # train
    for epoch in tqdm(range(1, cfg.niter + cfg.niter_decay + 1)):
        
        trainer = iter(train_data)

        ######################
        # Training
        ######################
        for iteration, batch in enumerate(trainer):
            net_g.train()
            torch.set_grad_enabled(True)
            # forward
            input_src, mask_true = batch[0].to(gpu), batch[1].to(gpu)
            mask_false = net_g(input_src)

            ######################
            # (1) Update D network
            ######################

            optimizer_d.zero_grad()
            
            # train with fake
            fake_pair = torch.cat((input_src, mask_false), 1)
            pred_fake = net_d.forward(fake_pair.detach())
            loss_d_fake = criterionGAN(pred_fake, False)

            # train with real
            real_pair = torch.cat((input_src, mask_true), 1)
            pred_real = net_d.forward(real_pair)
            loss_d_real = criterionGAN(pred_real, True)
            
            # Combined D loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            loss_d.backward()
        
            optimizer_d.step()

            ######################
            # (2) Update G network
            ######################

            optimizer_g.zero_grad()

            # First, G(A) should fake the discriminator
            fake_pair = torch.cat((input_src, mask_false), 1)
            pred_fake = net_d.forward(fake_pair)
            loss_g_gan = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_g_l1 = criterionL1(mask_false, mask_true) * cfg.lamb
            
            loss_g = loss_g_gan + loss_g_l1
            
            loss_g.backward()

            optimizer_g.step()

            ######################
            # Output Log 
            ######################
            if iteration % cfg.write_log_interval == 0 or iteration == 0:
                print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                    epoch, iteration, len(trainer), loss_d.item(), loss_g.item()))
                
            ######################
            # Save Image 
            ######################
            if iteration % cfg.snap_period == 0 or iteration == 0:

                savedir = os.path.join(cfg.train_result_dir, train_name, "images", str(epoch)+'-'+str(iteration))
            
                example_iter = iter(example_loader)
                net_g.eval()
                torch.set_grad_enabled(False)

                for ex_iter, batch in enumerate(example_iter):
            
                    i_s = batch[0].to(gpu)
                    name = str(batch[1][0])
                    
                    o_mask = net_g(i_s)
                    o_mask = o_mask.squeeze(0).to('cpu')
                    
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)

                    o_mask = F.to_pil_image(o_mask)
                    o_mask.save(os.path.join(savedir, name + '.png'))
            
            ######################
            # Save Weight 
            ######################
            if iteration % cfg.save_ckpt_interval == 0 or iteration == 0:

                savedir = os.path.join(cfg.train_result_dir, train_name, "weight")

                if not os.path.exists(savedir):
                    os.makedirs(savedir)
         
                net_g_model_out_path = savedir + "/netG_model_{}.pth".format(str(epoch)+'-'+str(iteration))
                net_d_model_out_path = savedir + "/netD_model_{}.pth".format(str(epoch)+'-'+str(iteration))
                torch.save(net_g.state_dict(), net_g_model_out_path)
                torch.save(net_d.state_dict(), net_d_model_out_path)

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

            
        
if __name__ == '__main__':
    main()