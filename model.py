import torch
from torch import nn
import numpy as np
import math

class CausalPad1d(nn.modules.Module):
    def __init__(self, kernel_size, dilation, inplace: bool = False):
        super(CausalPad1d, self).__init__()
        left_pad = ((kernel_size - 1) * dilation, 0)
        self.padder = nn.ConstantPad1d(left_pad, 0)
        self.inplace = inplace

    def forward(self, tensor):
        result = self.padder(tensor)
        return result

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
    

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def prior_snr(mag_clean, mag_noise):
    """Instantaneous a priori SNR in dB between speech and noise spectrums."""
    P_c = mag_clean ** 2
    P_n = mag_noise ** 2
    eps = torch.ones_like(mag_clean)
    eps = eps * 1e-12
    snr_db = 10 * torch.log10(torch.where(P_c / torch.where(P_n < 1e-12, eps, P_n) < 1e-12, eps, P_c))

    return snr_db

def xi_bar(mag_clean, mag_noise, mu, sigma):
    """
    Mapped a priori SNR in dB from prior snr.

    Argument/s:
        mag_clean - clean-speech short-time magnitude spectrum.
        mag_noise - noise short-time magnitude spectrum.

    Returns:
        Mapped a priori SNR(value in [0, 1]) in dB.
    """
    snr_db = prior_snr(mag_clean, mag_noise)
    snr_mapped = torch.zeros_like(snr_db)

    for i in range(snr_db.shape[0]):  # frequency bin
        for j in range(snr_db.shape[1]):  # time index
            snr_mapped[i, j] = \
                0.5 * (1 + torch.erf((snr_db[i, j] - mu[i]) / (sigma[i] * math.sqrt(2.0))))

    return snr_mapped


class Tcn(nn.Module):
    def __init__(
            self,
            input_size,
            n_output,
            n_blocks,
            d_model,
            d_f,
            k,
            max_d_rate):
        super(Tcn, self).__init__()
        self.d_model = d_model
        self.d_f = d_f
        self.k = k
        self.n_output = n_output
        self.n_blocks = n_blocks
        self.input_size = input_size
        self.first_layer = nn.Sequential(
            nn.Conv1d(self.input_size, self.d_model, 1, dilation=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(self.d_model, eps=1e-6),
        )
        self.block_layer = nn.ModuleList()
        block_in = d_model
        for i in range(n_blocks):
            self.block_layer.append(self.block(block_in, int(2 ** (i % (np.log2(max_d_rate) + 1)))))
        self.last_layer = nn.Sequential(
            nn.Conv1d(self.d_model, self.n_output, 1, dilation=1, bias=True)
        )

    def block(self, input_size, d_rate):
        """:Bottleneck residual block
        """
        conv_1 = self.unit(input_size, self.d_f, 1, 1)
        conv_2 = self.unit(self.d_f, self.d_f, self.k, d_rate)
        conv_3 = self.unit(self.d_f, self.d_model, 1, 1)
        return nn.Sequential(
            conv_1,
            conv_2,
            conv_3
        )

    def unit(self, input_size, n_filt, k, d_rate):
        """:Convolution unit
        """
        return nn.Sequential(
            nn.BatchNorm1d(input_size, eps=1e-6),
            nn.ReLU(),
            CasualPad1D(k, d_rate),  # padding to same
            nn.Conv1d(in_channels=input_size,
                      out_channels=n_filt,
                      kernel_size=k,
                      padding=0,  
                      bias=True,
                      dilation=d_rate)
        )

    def forward(self, in_feature):
        first_layer_out = self.first_layer(in_feature)
        block_in = first_layer_out
        for i in range(self.n_blocks):
            block_out = self.block_layer[i](block_in)
            residual = block_in + block_out
            block_in = residual
        output = self.last_layer(residual)
        return output


class Cnns(nn.Module):
    def __init__(self,
                 encoder_channel_size=[32, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256],
                 kernel_size=3,
                 padding=1,
                 dropout=0.2,
                 frame_len=512,
                 frame_shift=128,
                 device='cuda:0'):
        super(Cnns, self).__init__()
        self.encoder_channel_size, self.decoder_channel_size = encoder_channel_size[:], encoder_channel_size[:]
        self.decoder_channel_size.reverse()  # mirror
        self.dropout = dropout  # dropout is applied every three layers
        self.kernel_size = kernel_size
        self.padding = padding
        self.frame_len = frame_len
        self.frame_shift = frame_shift
        self.n_fft = frame_len
        self.device = device
        self.window = torch.hann_window(self.frame_len).cuda(self.device)
        """
        self.real_mtx = torch.zeros(self.n_fft, self.frame_len).cuda(self.device)  # D_r
        self.imag_mtx = torch.zeros(self.n_fft, self.frame_len).cuda(self.device)  # D_i
        for k in range(self.n_fft):
            self.real_mtx[:, k] = torch.cos(torch.arange(0, self.frame_len) * k * 2 * np.pi / self.frame_len)
            self.imag_mtx[:, k] = torch.sin(torch.arange(0, self.frame_len) * k * 2 * np.pi / self.frame_len)
        self.window = torch.hann_window(self.frame_len).cuda(self.device)
        self.mag_out = torch.zeros(32, self.n_fft // 2 + 1, 125, requires_grad=True).cuda(self.device)
        """

        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.encoder_channel_size[0], kernel_size=1),
            nn.Tanh()
        )

        self.encoder = nn.ModuleList()
        """encode part, reduce size and expand channels"""

        for idx in range(len(self.encoder_channel_size) - 1):  # encoder
            layer = nn.Sequential()
            if idx % 3 == 0:
                layer.add_module('dropout', nn.Dropout(self.dropout))
            layer.add_module('conv_' + str(idx + 1), nn.Conv1d(in_channels=self.encoder_channel_size[idx],
                                                               out_channels=self.encoder_channel_size[idx + 1],
                                                               kernel_size=self.kernel_size, stride=2,
                                                               padding=self.padding))
            layer.add_module('norm', nn.BatchNorm1d(encoder_channel_size[idx + 1]))
            layer.add_module('relu', nn.ReLU())
            self.encoder.append(layer)

        self.decoder = nn.ModuleList()
        """decode part, expand size and reduce channels, mirror with encode"""

        for idx in range(len(self.decoder_channel_size) - 1):  # decoder
            layer = nn.Sequential()
            if idx == 0:
                double = 1
            else:
                double = 2
            if idx % 3 == 0:
                layer.add_module('dropout', nn.Dropout(self.dropout))
            layer.add_module('conv_' + str(idx + 1), nn.Conv1d(in_channels=self.decoder_channel_size[idx] * double,
                                                               # cat from encode, so double the input channel
                                                               out_channels=self.decoder_channel_size[idx + 1],
                                                               kernel_size=self.kernel_size, stride=1))
            layer.add_module('norm', nn.BatchNorm1d(self.decoder_channel_size[idx + 1]))
            layer.add_module('relu', nn.ReLU())
            self.decoder.append(layer)

        self.last_layer = nn.Sequential(
            nn.Conv1d(in_channels=self.decoder_channel_size[-1] * 2, out_channels=1, kernel_size=1),
            nn.Tanh()
        )
        self.weight_init()

    def forward(self, inputs):
        skip = []
        layer_in = self.first_layer(inputs)
        skip.append(layer_in)

        """encode forward part"""
        for layer_idx, encode_layer in enumerate(self.encoder):
            skip.append(layer_in)
            layer_out = encode_layer(layer_in)
            layer_in = layer_out

        """decode forward part"""
        for layer_idx, decode_layer in enumerate(self.decoder):
            """insert zeros between the consecutive samples to up sample"""
            expand_layer = torch.zeros((layer_in.shape[0], layer_in.shape[1],
                                        layer_in.shape[2] * 2 - 1 + self.padding * 2 + 1)).cuda(self.device)
            expand_layer[:, :, [i for i in range(self.padding, expand_layer.shape[2] - self.padding - 1, 2)]] \
                = layer_in
            layer_out = decode_layer(expand_layer)
            layer_out = torch.cat((layer_out, skip[-layer_idx - 1]),
                                  dim=1)  # skip cat in channel dim
            layer_in = layer_out

        wave_out = self.last_layer(layer_in)
        mag_out = self.wave2spec(wave_out)  # convert wave to mag spec using torch.stft
        return wave_out, mag_out

    def wave2spec(self, wave_out):
        spec_out = torch.stft(wave_out.squeeze(), n_fft=self.n_fft, hop_length=self.frame_shift,
                              win_length=self.frame_len, window=self.window,
                              center=False).cuda(self.device)
        # mag = sqrt(real_part ** 2 + imag_part ** 2)
        mag_out = torch.sqrt(torch.sum(spec_out ** 2, -1) + 1e-12)
        return mag_out

    def snr_mapped(self, mag_speech, mag_noise, mu, sigma):
        return xi_bar(mag_speech, mag_noise, mu, sigma)

    def weight_init(self):
        for child in self.first_layer:
            if isinstance(child, nn.Conv1d):
                nn.init.xavier_normal_(child.weight)
        for module in self.encoder:
            for child in module.children():
                if isinstance(child, nn.Conv1d):
                    nn.init.xavier_normal_(child.weight)
        for module in self.decoder:
            for child in module.children():
                if isinstance(child, nn.Conv1d):
                    nn.init.xavier_normal_(child.weight)
        for child in self.first_layer:
            if isinstance(child, nn.Conv1d):
                nn.init.xavier_normal_(child.weight)

    def out2mag(self, outputs):
        """using transform matrix to convert output to magnitude while keeping it differential"""
        n_frames = (outputs.shape[-1] - self.frame_len) // self.frame_shift + 1
        for sequence in range(outputs.shape[0]):
            for frame_idx in range(n_frames):
                frame = outputs[sequence, :, frame_idx * self.frame_shift:frame_idx * self.frame_shift + self.frame_len]
                frame = frame * self.window
                real_part = torch.mm(frame.view(1, -1), self.real_mtx).squeeze()
                imag_part = torch.mm(frame.view(1, -1), self.imag_mtx).squeeze()
                real_stft = real_part[:self.n_fft // 2 + 1]
                imag_stft = imag_part[:self.n_fft // 2 + 1]
                self.mag_out[sequence, :, frame_idx] = torch.sqrt(
                    real_stft ** 2 + imag_stft ** 2 + 1e-12)  # get magnitude


class MyCnns(nn.Module):
    def __init__(self,
                 encoder_channel_size=[512, 256, 128, 64, 32, 16, 8],
                 dropout=0.2,
                 kernel_size=3,
                 max_dilation=32):
        super(MyCnns, self).__init__()
        self.encoder_channel_size = encoder_channel_size[:]
        self.decoder_channel_size = encoder_channel_size[:]
        self.decoder_channel_size.reverse()
        self.dropout = dropout
        self.kernel_size = 3
        self.max_dilation = max_dilation

        self.encoder = nn.Sequential()
        for k in range(self.encoder_channel_size):
            input_channel = self.encoder_channel_size[k]
            output_channel = self.encoder_channel_size[k+1]
            self.encoder.add_module('block' + str(k), self.block(input_channel, output_channel, max_dilation=16))

        self.decoder = nn.Sequential()
        for k in range(self.decoder_channel_size):
            input_channel = self.decoder_channel_size[k]
            output_channel = self.decoder_channel_size[k+1]
            self.decoder.add_module('block' + str(k) ,self.block(input_channel, output_channel, max_dilation=16))

    def forward(self, inputs):
        print('wang xu zhen de shi xiao zhu')

    def block(self, input_channel, output_channel, dilation):
        """:Bottleneck residual block
        """
        mid_channel = (input_channel + output_channel) // 2
        unit_1 = self.unit(input_channel, mid_channel, 1, 1)
        unit_2 = self.unit(mid_channel, mid_channel, self.kernel_size, dilation)
        unit_3 = self.unit(mid_channel, output_channel, 1, 1)
        return nn.Sequential(
            unit_1,
            unit_2,
            unit_3
        )

    def unit(self, input_channel, output_channel, kernel_size, dilation):
        """:Convolution unit
        """
        return nn.Sequential(
            nn.BatchNorm1d(input_channel, eps=1e-6),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_channel,
                      out_channels=output_channel,
                      kernel_size=kernel_size,
                      padding=dilation * (kernel_size - 1) // 2,  # padding to same
                      bias=True,
                      dilation=dilation)
        )


if __name__ == "__main__":
    device = 'cuda:0'
    in_features = torch.randn(32, 257, 100).cuda(device)
    # in_features = torch.randn(32, 1, 16384).cuda(device)
    # net = Cnns(kernel_size=11, padding=5).cuda(device)
    # print('wang xu shi xiao zhu')
    net = Tcn(input_size=257,
              n_output=257,
              d_model=257,
              k=3,
              max_d_rate=16,
              n_blocks=40,
              d_f=64).cuda(device)

    net_info = get_parameter_number(net)
    print(net_info)
    out = net(in_features)
    # out, mag = net(in_features)
    print(out.shape)
    # print(mag.shape)
    print('wang xu shi xiao zhu ba')
