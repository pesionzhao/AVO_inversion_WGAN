"""
@File    : train.py
@Author  : Pesion
@Date    : 2023/11/1
@Desc    : 
"""
import sys

import numpy as np

sys.path.append('..')
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from random import random

from torch.utils.data import DataLoader
from Model import Feature_extraction, Discriminator, ForWard
import torch
import torch.nn as nn
from Gan_DataSet import Marmousi2FromMat, WGAN_Data
from util.DataSet import Set_origin
from util.utils import read_yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

record = True
resume = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_path = 'cfg/unet.yaml'
cfg = read_yaml(config_path=config_path)
cfg = argparse.Namespace(**cfg)
train_dateset = WGAN_Data(data_path=cfg.dataset_path, traces=cfg.dataset_traces,
                          train_traces=cfg.dataset_usetraces)
train_dataloader = DataLoader(dataset=train_dateset, batch_size=cfg.batchsize, num_workers=0)

G = Feature_extraction().to(device)
D_vp = Discriminator().to(device)
D_vs = Discriminator().to(device)
D_rho = Discriminator().to(device)
F = ForWard().to(device)
if resume:
    F.load_state_dict(torch.load('weights/F.pth', map_location='cpu'))
    G.load_state_dict(torch.load('weights/G.pth', map_location='cpu'))
    D_vp.load_state_dict(torch.load('weights/D_vp.pth', map_location='cpu'))
    D_vs.load_state_dict(torch.load('weights/D_vs.pth', map_location='cpu'))
    D_rho.load_state_dict(torch.load('weights/D_rho.pth', map_location='cpu'))
# F.load_state_dict(torch.load('weightsnew/F.pth', map_location='cpu'))
vp_optimizer = torch.optim.Adam(D_vp.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.9))
vs_optimizer = torch.optim.Adam(D_vs.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.9))
rho_optimizer = torch.optim.Adam(D_rho.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.9))
G_optimizer = torch.optim.Adam(G.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.9))
F_optimizer = torch.optim.Adam(F.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.9))
# vp_optimizer = torch.optim.RMSprop(D_vp.parameters(), lr=cfg.learning_rate)
# vs_optimizer = torch.optim.RMSprop(D_vs.parameters(), lr=cfg.learning_rate)
# rho_optimizer = torch.optim.RMSprop(D_rho.parameters(), lr=cfg.learning_rate)
# G_optimizer = torch.optim.RMSprop(G.parameters(), lr=cfg.learning_rate)
# F_optimizer = torch.optim.RMSprop(F.parameters(), lr=cfg.learning_rate)
G_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=0.99)
F_scheduler = torch.optim.lr_scheduler.ExponentialLR(F_optimizer, gamma=0.99)
G_loss = nn.L1Loss()
alpha = 50
beta = 1500
lam = 10
savepath = './weights'
one = torch.tensor(1, dtype=torch.float, requires_grad=True).to(device)
mone = -1 * one

settings = Set_origin(layers=cfg.input_len, dt0=cfg.wavelet['dt0'], wave_f=cfg.wavelet['f0'], wave_n0=cfg.wavelet['n0'],
                      theta_list=None)
settings.setup()


def calculate_gradient_penalty(discriminator, real_images, fake_images, m_back):  # label and generate param
    eta = torch.FloatTensor(1, 1, 1).uniform_()
    eta = eta.expand(1, real_images.size(1), real_images.size(2)).to(device)
    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.to(device)

    # define it to calculate gradient
    interpolated.requires_grad_()

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(torch.cat((interpolated, m_back), dim=1))

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(
                                        prob_interpolated.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    grad_penalty = ((gradients.norm(2) - 1) ** 2).mean() * lam
    return grad_penalty

if record:
    writer = SummaryWriter("log4")
for i in range(cfg.epochs):
    best_loss = 1000
    total_loss = 0
    D_vp_loss = 0
    D_vs_loss = 0
    D_rho_loss = 0
    close_loss = 0
    open_loss = 0
    g_loss = 0
    train_D_bar = tqdm(train_dataloader)
    for _, (m_label, angle_gather, m_back) in enumerate(train_D_bar):
        D_vp.train()
        D_vs.train()
        D_rho.train()
        G.train()
        F.train()
        G.zero_grad()
        F.zero_grad()
        for p in D_vp.parameters():
            p.requires_grad = True
        for p in D_vs.parameters():
            p.requires_grad = True
        for p in D_rho.parameters():
            p.requires_grad = True
        D_vp.zero_grad()
        D_vs.zero_grad()
        D_rho.zero_grad()
        # 遇到的问题有
        # 1. 每次反向传播后梯度清空，使得G_out无法用作下一次的更新，retain_graph=True得以解决但还需要深究
        # 2. gp的计算没有使用梯度，需要解决！
        # 3. GAN网络的训练流程，我这样子写是否合理
        # 4. 判别器的损失和L_g都是负数！！！论文里判别器在前五轮损失就趋近于零，是否需要加Sigmoid

        # train discriminator
        m_label, angle_gather, m_back = m_label[0:1, ...].to(device), angle_gather.to(
            device), m_back.to(
            device)
        G_out = G(angle_gather[0:1], m_back[0:1, 0:1, :], m_back[0:1, 1:2, :], m_back[0:1, 2:3, :])
        # writer.add_graph(G,[angle_gather, m_back])
        # vp_optimizer.zero_grad()
        input1, input2, input3 = torch.split(G_out, 1, dim=1)
        vp_input = torch.cat((input1, m_back[0:1, 0:1, :]), dim=1)
        fake_vp_loss = D_vp(vp_input)

        # fake_vp_loss = D_vp(G_out[0:1, 0:1, :], m_back[0:1, 0:1, :])
        fake_vp_loss = fake_vp_loss.mean()
        # fake_vp_loss.backward(one, retain_graph=True)

        real_vp_loss = D_vp(torch.cat((m_label[:, 0:1, :], m_back[0:1, 0:1, :]), dim=1))
        real_vp_loss = real_vp_loss.mean()
        # real_vp_loss.backward(mone)

        gpp = calculate_gradient_penalty(D_vp, m_label[:, 0:1, :], input1, m_back[0:1, 0:1, :])
        # gpp.backward(retain_graph=True)
        L_vp_D = fake_vp_loss - real_vp_loss + gpp
        # L_vp_D.backward()
        L_vp_D.backward(retain_graph=True)

        # L_vp_D = torch.sum(
        #     D_vp(G_out[0:1, 0:1, :], m_back[0:1, 0:1, :]) - D_vp(m_label[:, 0:1, :], m_back[0:1, 0:1, :])) + gpv
        # L_vp_D.backward(retain_graph=True)
        vp_optimizer.step()

        # G_out = G(angle_gather, m_back[:,0:1,:],m_back[:,1:2,:],m_back[:,2:3,:])

        # vs_optimizer.zero_grad()
        # D_vs_out = D_vs(m_label[:, 1:2, :], m_back[:, 1:2, :])
        # gps = (torch.norm(D_vs(ratio * m_label[:, 1:2, :] + (1 - ratio) * G_out[:, 1:2, :], m_back[:, 1:2, :]),
        #                   p=2) - 1) ** 2
        fake_vs_loss = D_vs(torch.cat((input2, m_back[0:1, 1:2, :]), dim=1))
        fake_vs_loss = fake_vs_loss.mean()
        # fake_vs_loss.backward(one, retain_graph=True)

        real_vs_loss = D_vs(torch.cat((m_label[:, 1:2, :], m_back[0:1, 1:2, :]), dim=1))
        real_vs_loss = real_vs_loss.mean()
        # real_vs_loss.backward(mone)

        gps = calculate_gradient_penalty(D_vs, m_label[:, 1:2, :], input2, m_back[0:1, 1:2, :])
        # gps.backward(retain_graph=True)
        L_vs_D = fake_vs_loss - real_vs_loss + gps
        # L_vs_D.backward()
        L_vs_D.backward(retain_graph=True)
        vs_optimizer.step()

        # G_out = G(angle_gather, m_back[:,0:1,:],m_back[:,1:2,:],m_back[:,2:3,:])

        # rho_optimizer.zero_grad()
        # D_rho_out = D_rho(m_label[:, 2:3, :], m_back[:, 2:3, :])
        fake_rho_loss = D_rho(torch.cat((input3, m_back[0:1, 2:3, :]), dim=1))
        fake_rho_loss = fake_rho_loss.mean()
        # fake_rho_loss.backward(one, retain_graph=True)

        real_rho_loss = D_rho(torch.cat((m_label[:, 2:3, :], m_back[0:1, 2:3, :]), dim=1))
        real_rho_loss = real_rho_loss.mean()
        # real_rho_loss.backward(mone)

        gprho = calculate_gradient_penalty(D_rho, m_label[:, 2:3, :], input3, m_back[0:1, 2:3, :])
        # gprho.backward()
        L_rho_D = fake_rho_loss - real_rho_loss + gprho
        L_rho_D.backward()

        # L_rho_D.backward(retain_graph=False)
        rho_optimizer.step()

        train_D_bar.desc = f"train epoch[{i + 1}/{cfg.epochs}] " \
                           f"loss: D_vp:{L_vp_D:.2f}, " \
                           f"D_vs:{L_vs_D:.2f}, " \
                           f"D_rho:{L_rho_D:.2f}, "
        D_vp_loss = D_vp_loss + L_vp_D
        D_vs_loss = D_vs_loss + L_vs_D
        D_rho_loss = D_rho_loss + L_rho_D

    if record:
        writer.add_scalar('Dvp_loss', D_vp_loss / len(train_D_bar), i)
        writer.add_scalar('Dvs_loss', D_vs_loss / len(train_D_bar), i)
        writer.add_scalar('Drho_loss', D_rho_loss / len(train_D_bar), i)

    train_G_bar = tqdm(train_dataloader)
    for _, (m_label, angle_gather, m_back) in enumerate(train_G_bar):
        for p in D_vp.parameters():
            p.requires_grad = False  # to avoid computation
        for p in D_vs.parameters():
            p.requires_grad = False  # to avoid computation
        for p in D_rho.parameters():
            p.requires_grad = False  # to avoid computation

        G_optimizer.zero_grad()
        F_optimizer.zero_grad()

        # train generator
        m_label, angle_gather, m_back = m_label[0:1, ...].to(device), angle_gather.to(
            device), m_back.to(
            device)
        G_out = G(angle_gather, m_back[:, 0:1, :], m_back[:, 1:2, :], m_back[:, 2:3, :])
        F_out = F(m_label)

        L_g = (-D_vp(torch.cat((G_out[0:1, 0:1, :], m_back[0:1, 0:1, :]), dim=1))
               - D_vs(torch.cat((G_out[0:1, 1:2, :], m_back[0:1, 1:2, :]), dim=1))
               - D_rho(torch.cat((G_out[0:1, 2:3, :], m_back[0:1, 2:3, :]), dim=1))).mean()

        # L_g.backward(retain_graph=True)

        L_open_1 = G_loss(G_out[0:1, :, :], m_label[:, :, :])

        L_open_2 = G_loss(F_out, angle_gather[0:1, ...])

        L_close_1 = G_loss(F(G_out), angle_gather)
        L_close_2 = G_loss(G(F_out, m_back[0:1, 0:1, :], m_back[0:1, 1:2, :], m_back[0:1, 2:3, :]), m_label)
        L_GF = L_g + beta * (L_open_1 + L_open_1) + alpha * (L_close_1 + L_close_2)

        L_GF.backward()

        G_optimizer.step()
        F_optimizer.step()
        total_loss += L_GF
        close_loss = close_loss + L_close_1 + L_close_2
        open_loss = open_loss + L_open_1 + L_open_2
        g_loss = g_loss + L_g

        train_G_bar.desc = f"train epoch[{i + 1}/{cfg.epochs}] " \
                           f"L_g:{L_g:.2f}" \
                           f"L_open:{L_open_1 + L_open_2:.2f}" \
                           f"L_close:{L_close_1 + L_close_2:.2f}" \
                           f"G&F:{L_GF:.2f} "
    G_scheduler.step()
    F_scheduler.step()
    if record:
        writer.add_scalar('open_loss', open_loss / len(train_G_bar), i)
        writer.add_scalar('close_loss', close_loss / len(train_G_bar), i)
        writer.add_scalar('g_loss', g_loss / len(train_G_bar), i)
        writer.add_scalar('G&F_loss', total_loss / len(train_G_bar), i)
    print(f'total loss = {total_loss / len(train_G_bar):.3f}')

    if (total_loss / 100) < best_loss:
        torch.save(G.state_dict(), os.path.join(savepath, 'G.pth'))
        torch.save(F.state_dict(), os.path.join(savepath, 'F.pth'))
        torch.save(D_vp.state_dict(), os.path.join(savepath, 'D_vp.pth'))
        torch.save(D_vs.state_dict(), os.path.join(savepath, 'D_vs.pth'))
        torch.save(D_rho.state_dict(), os.path.join(savepath, 'D_rho.pth'))
