# -*- coding: utf-8 -*-
# @Author: xin zhang
# @Date:   2020/7/23 14:38
# @Last Modified by:   xin zhang
# @Last Modified time: 2020/7/23 14:38
import torch
from torch import nn, optim
from model import Tcn, Cnns
from dataset import MyDataset
from torch.utils.data import DataLoader
import utils
import numpy as np


# speech_dir = '/home/dcase/c2019/zx/se/deepxi/set/train_clean_speech/'
speech_dir = '/home/dcase/c2019/zx/datasets/clean_testset_wav_48k/'
noise_dir = '/home/dcase/c2019/zx/datasets/noisex-92-16k/'
device = 'cuda:1'
train_speech_dir = speech_dir
train_noise_dir = noise_dir
sr = 16000
frame_len = 512
frame_shift = 128
n_fft = 512
block_size = 32  # length of one sequence is block_size x frame_len, so it's 16384
snr_level = [i for i in range(-5, 0)]
train_dataset = MyDataset(speech_dir=train_speech_dir, noise_dir=train_noise_dir,
                          sr=sr, snr_level=snr_level, block_size=block_size,
                          frame_len=frame_len, frame_shift=frame_shift)
data_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4, drop_last=True)
print(data_loader.__len__())
model_path = '.model_checkpoints/'

"""
net = Tcn(input_size=257, k=3, d_f=64, d_model=257, n_output=257, n_blocks=5, max_d_rate=8).cuda(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
epochs = 10
loss = nn.MSELoss()

for epoch in range(epochs):
    index = 0
    for mag_x, pha_x, _, _, mask_x in data_loader:
        mag_x = mag_x.cuda(device)
        mask_x = mask_x.cuda(device)
        # forward and back prop
        mask_y = net(mag_x)

        cost = loss(mask_x, mask_y)
        optimizer.zero_grad()

        cost.backward()
        # update model parameters
        optimizer.step()
        print(index, cost)
        index += 1
        # print(mask_y)


"""
# train
net = Cnns(kernel_size=11, padding=5, device=device).cuda(device)
optimizer = optim.Adam(net.parameters(), lr=0.0002)
num_epoch = 10
loss = nn.MSELoss()
# net.apply(weight_init)
for epoch in range(num_epoch):
    batch = 0
    for mag_noisy, pha_noisy, wave_speech, wave_noisy, snr_db in data_loader:
        wave_noisy = wave_noisy.cuda(device)
        mag_noisy = mag_noisy.cuda(device)
        # predict
        wave_hat, mag_hat = net(wave_noisy)
        # loss between noisy and hat magnitude spectrums
        cost = loss(mag_noisy, mag_hat)
        # clear gradient
        optimizer.zero_grad()
        # gradient back prop
        cost.backward()
        # update model parameters
        optimizer.step()
        print('=========> epoch : %d, batch : %d, loss : %f' % (epoch, batch, cost))
        batch += 1
        if batch % 30 == 0:
            utils.specgrum(wave_noisy[0].cpu().numpy().squeeze(), frame_len, frame_shift, n_fft,
                           title='Noisy Spectrum'+str(batch))
            utils.specgrum(wave_hat[0].detach().cpu().numpy().squeeze(), frame_len, frame_shift, n_fft,
                           title='Hat Spectrum'+str(batch))


torch.save(net.state_dict(), 'timit_noisex92_model.pkl')

"""

"""
# inference
noisy_dir = ''
noisy_files = utils.find_files(noisy_dir)
net = Cnns(kernel_size=11, padding=5)
model_dict_path = '.model_checkpoints/timit_noisex92_model.pkl'
net.load_state_dict(model_dict_path)
