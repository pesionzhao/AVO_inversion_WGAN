"""
@File    : Gan_DataSet.py
@Author  : Pesion
@Date    : 2023/11/1
@Desc    : 
"""
import json
import os

import numpy as np
import scipy.io as scio

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import yaml


class Marmousi2FromMat(Dataset):
    def __init__(self, data_path, traces, train_traces: dict):
        super(Marmousi2FromMat, self).__init__()
        self.Train_Data = scio.loadmat(data_path)
        self.traces = traces
        self.start = train_traces['start'] if train_traces['start'] is not None else 0
        self.stop = train_traces['stop'] if train_traces['stop'] is not None else self.traces
        self.step = train_traces['step'] if train_traces['step'] is not None else 1

    def __len__(self):
        return int((self.stop - self.start) / self.step)

    def __getitem__(self, index):  # torch.tensor改为torch.from_numpy
        # 这里的vp vs rho 大小为[layers, traces]
        # 三参数标签
        index = index*self.step + self.start
        vp_label = torch.tensor(self.Train_Data['vp'][..., index], dtype=torch.float32)
        vs_label = torch.tensor(self.Train_Data['vs'][..., index], dtype=torch.float32)
        den_label = torch.tensor(self.Train_Data['rho'][..., index], dtype=torch.float32)
        M_sample = torch.stack([vp_label, vs_label, den_label], dim=0)

        # 地震数据 -> [Trace, 3, input_len]
        Seis_low = torch.tensor(self.Train_Data['S_low'][index], dtype=torch.float32)
        Seis_mid = torch.tensor(self.Train_Data['S_mid'][index], dtype=torch.float32)
        Seis_large = torch.tensor(self.Train_Data['S_large'][index], dtype=torch.float32)
        S_sample = torch.stack([Seis_low, Seis_mid, Seis_large], dim=0)

        # 低频三参数数据 -> [Trace, 3, input_len]
        vp_sample_initial = torch.tensor(self.Train_Data['vp_low'][..., index], dtype=torch.float32)
        vs_sample_initial = torch.tensor(self.Train_Data['vs_low'][..., index], dtype=torch.float32)
        den_sample_initial = torch.tensor(self.Train_Data['rho_low'][..., index], dtype=torch.float32)
        M_sample_initial = torch.stack([vp_sample_initial, vs_sample_initial, den_sample_initial], dim=0)

        return M_sample, S_sample, M_sample_initial

class WGAN_Data(Dataset):
    def __init__(self, data_path, traces, train_traces: dict):
        super(WGAN_Data, self).__init__()
        self.raw_Data = scio.loadmat(data_path)
        self.Train_Data = {}
        self.traces = traces
        self.start = train_traces['start'] if train_traces['start'] is not None else 0
        self.stop = train_traces['stop'] if train_traces['stop'] is not None else self.traces
        self.step = train_traces['step'] if train_traces['step'] is not None else 1
        self.normalize()


    def __len__(self):
        return int((self.stop - self.start) / self.step)

    def __getitem__(self, index):  # torch.tensor改为torch.from_numpy
        # 这里的vp vs rho 大小为[layers, traces]
        # 三参数标签
        index = index*self.step + self.start
        vp_label = torch.tensor(self.Train_Data['vp'][..., index], dtype=torch.float32)
        vs_label = torch.tensor(self.Train_Data['vs'][..., index], dtype=torch.float32)
        den_label = torch.tensor(self.Train_Data['rho'][..., index], dtype=torch.float32)
        M_sample = torch.stack([vp_label, vs_label, den_label], dim=0)

        # 地震数据 -> [Trace, 3, input_len]
        angle_gahter = torch.tensor(self.Train_Data['S'][index].swapaxes(-1,-2), dtype=torch.float32)

        # 低频三参数数据 -> [Trace, 3, input_len]
        vp_sample_initial = torch.tensor(self.Train_Data['vp_low'][..., index], dtype=torch.float32)
        vs_sample_initial = torch.tensor(self.Train_Data['vs_low'][..., index], dtype=torch.float32)
        den_sample_initial = torch.tensor(self.Train_Data['den_low'][..., index], dtype=torch.float32)
        M_sample_initial = torch.stack([vp_sample_initial, vs_sample_initial, den_sample_initial], dim=0)

        return M_sample, angle_gahter, M_sample_initial

    def normalize(self):
        param = ['vp', 'vs', 'rho', 'S', 'vp_low', 'vs_low', 'den_low']
        if os.path.exists('normalize_dict.json'):
            with open('normalize_dict.json','r') as json_file:
                normalize_dict = json.load(json_file)
            for i in param:
                if i == 'xx':
                    imax = normalize_dict[i]
                    self.Train_Data[i] = self.raw_Data[i] / imax
                else:
                    imin = normalize_dict[i]['min']
                    imax = normalize_dict[i]['max']
                    self.Train_Data[i] = (self.raw_Data[i] - imin) / (imax - imin)
        else:
            normalize_dict = {}
            for i in param:
                if i == 'xx':
                    imax = np.max(self.raw_Data[i])
                    self.Train_Data[i] = self.raw_Data[i] / imax
                    normalize_dict.update({f"{i}": imax})
                else:
                    imin = np.min(self.raw_Data[i])
                    imax = np.max(self.raw_Data[i])
                    self.Train_Data[i] = (self.raw_Data[i] - imin) / (imax - imin)
                    normalize_dict.update({f"{i}": {"max": imax, "min": imin}})
            with open('normalize_dict.json','w') as json_file:
                json.dump(normalize_dict, json_file)
