# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import os
import random
import time
import warnings

warnings.filterwarnings("ignore") 
from math import log10

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lpips import lpips
from pytorch_msssim import SSIM as pytorch_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_training_set
from image_utils import TVLoss
from laploss import LapLoss
from model_x import *
from network import decoder3, decoder4, encoder3, encoder4





# Training settings
parser = argparse.ArgumentParser(description='Example')
parser.add_argument('--data_dir', type=str, default="D:/Users/CVAE/datasets/STATES_x4_3D")
parser.add_argument('--up_factor', type=int, default=4, help='upsampling factor')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--model_type', type=str, default='GAN')
parser.add_argument('--patch_size', type=int, default=8, help='Size of cropped LR image')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='D:/Users/CVAE/states_models_x4_3D_x1/', help='Location to save checkpoint models')


opt = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def train(epoch):
    G_epoch_loss = 0
    D_epoch_loss = 0
    G.train()
    D.train()

    
    for iteration, batch in enumerate(training_data_loader):
        input, target, ref = batch['LR'], batch['HR'], batch['Ref']
        # ref = ref[0]
        minibatch = input.size()[0]
        real_label = torch.ones((minibatch, 7917))
        fake_label = torch.zeros((minibatch, 7917))

        input = input.to(device)
        target = target.to(device)



        ref = ref.to(device)
        real_label = real_label.to(device)
        fake_label = fake_label.to(device)

        # Reset gradient
        for p in D.parameters():
            p.requires_grad = False

        G_optimizer.zero_grad()

        # encoder
        
        bic = F.interpolate(input, scale_factor=opt.up_factor, mode='bilinear')
        


        ref_feat = enc(target)

        LR_feat = enc(bic)


        predict, KL = G(input,target, LR_feat['r41'], ref_feat['r41'])


        
        pre_LR = F.interpolate(predict, scale_factor=1.0 / opt.up_factor, mode='bilinear')

        LR_loss = L1_criterion(pre_LR, input)



       





        SR_L1 = L1_criterion(predict, target) + \
                L1_criterion(F.interpolate(predict, scale_factor=0.5, mode='bilinear'), F.interpolate(target, scale_factor=0.5, mode='bilinear')) + \
                L1_criterion(F.interpolate(predict, scale_factor=0.25, mode='bilinear'), F.interpolate(target, scale_factor=0.25, mode='bilinear'))


        ssim_loss = 1 - ssim(predict, target)
        lap_recon = lap_loss(predict, target)

        ContentLoss, StyleLoss = VGG_feat(predict, bic)
        lpips_sp = loss_fn_alex_sp(2 * predict - 1, 2 * target - 1)
        lpips_sp = lpips_sp.mean()
        D_fake_feat, D_fake_decision = D(predict)
        D_real_feat, D_real_decision = D(target)
        GAN_feat_loss = L1_criterion(D_fake_feat, D_real_feat)
        GAN_loss = L1_criterion(D_fake_decision, real_label)

        G_loss = 16*LR_loss + 1 * SR_L1 + 0.001 * ContentLoss + StyleLoss + 1e4*lpips_sp +1e4*ssim_loss+lap_recon+1e4*KL

        G_loss.backward()
        G_optimizer.step()

       
        G_epoch_loss += G_loss.data
        

        print("===> Epoch[{}]({}/{}): \
               G_loss: {:.4f} || "
              "LR_loss: {:.4f} || "
              "SR_L1: {:.4f} || "
              "GAN_loss: {:.4f} || "
              "ContentLoss: {:.4f} || "
              "StyleLoss: {:.4f} ||"
              "lpips_sp: {:.4f} ||"
              "GAN_feat_loss: {:.4f} ||".format(epoch, iteration,
                                          len(training_data_loader),
                                          G_loss.data,
                                          LR_loss.data,
                                          SR_L1.data,
                                          GAN_loss.data,
                                          ContentLoss.data,
                                          StyleLoss.data,
                                          lpips_sp.data,
                                          GAN_feat_loss.data))

    print("===> Epoch {} Complete: Avg. G Loss: {:.4f} || D Loss: {:.4f}".format(epoch, G_epoch_loss / len(training_data_loader),
                                                                                 D_epoch_loss / len(training_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    G_model_out_path = opt.save_folder + opt.model_type + "_generator_{}.pth".format(epoch)
    D_model_out_path = opt.save_folder + opt.model_type + "_discriminator_{}.pth".format(epoch)
    enc_model_out_path = opt.save_folder + opt.model_type + "_enc_{}.pth".format(epoch)
    dec_model_out_path = opt.save_folder + opt.model_type + "_dec_{}.pth".format(epoch)
    torch.save(G.state_dict(), G_model_out_path)
    torch.save(D.state_dict(), D_model_out_path)
    torch.save(enc.state_dict(), enc_model_out_path)
    torch.save(dec.state_dict(), dec_model_out_path)
    print("Checkpoint saved to {} and {}".format(G_model_out_path, D_model_out_path))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ =="__main__":


    setup_seed(123)
    print('===> Loading datasets')
    train_set = get_training_set(opt.data_dir, opt.patch_size, opt.up_factor,
                                opt.data_augmentation)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=True)

    print('===> Building model ', opt.model_type)


    enc = encoder4(1)
    dec = decoder4()
    G = VAE_v3_4x(up_factor=opt.up_factor)
    D = discriminator_v2(num_channels=1, base_filter=32)
    L1_criterion = nn.L1Loss(size_average=False)
    L2_criterion = nn.MSELoss(size_average=False)
    TV = TVLoss()
    ssim = pytorch_ssim()
    lap_loss = LapLoss(max_levels=5, k_size=5, sigma=2.0)

    VGG_feat = Vgg19_feat(device)
    loss_fn_alex_sp = lpips.LPIPS(spatial=True)

    print('---------- Generator architecture -------------')
    print_network(G)
    print('----------------------------------------------')
    print('---------- Discriminator architecture -------------')
    print_network(D)
    print('----------------------------------------------')


    enc_model_name = r"D:/Users/CVAE/models/vgg_r41.pth"
    if os.path.exists(enc_model_name):
        pretrained_dict = torch.load(enc_model_name, map_location=lambda storage, loc: storage)
        enc_model_dict = enc.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in enc_model_dict}
        enc_model_dict.update(pretrained_dict)
        enc.load_state_dict(enc_model_dict)
        print('encoder model is loaded!')

    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)


    if opt.pretrained:
        G_model_name = os.path.join(opt.save_folder + opt.pretrained_G_model)
        D_model_name = os.path.join(opt.save_folder + opt.pretrained_D_model)
        if os.path.exists(G_model_name):
            pretrained_dict = torch.load(G_model_name, map_location=lambda storage, loc: storage)
            model_dict = G.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            G.load_state_dict(model_dict)

            print('Pre-trained Generator is loaded.')
        if os.path.exists(D_model_name):
            D.load_state_dict(torch.load(D_model_name, map_location=lambda storage, loc: storage))
            print('Pre-trained Discriminator is loaded.')



    enc = enc.to(device)
    dec = dec.to(device)
    G = G.to(device)
    D = D.to(device)
    VGG_feat = VGG_feat.to(device)
    L1_criterion = L1_criterion.to(device)
    ssim = ssim.to(device)
    lap_loss = lap_loss.to(device)
    TV = TV.to(device)
    loss_fn_alex_sp = loss_fn_alex_sp.to(device)


    G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)


    for epoch in tqdm(range(opt.start_iter, opt.nEpochs + 1)):
        train(epoch)
        

        print(G_optimizer.param_groups[0]['lr'])
        if epoch % (opt.snapshots) == 0:
            checkpoint(epoch)
    print("end")
