import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import soundfile as sf
import librosa
import utils
import random


class MyDataset(Dataset):
    def __init__(self,
                 speech_dir,
                 noise_dir,
                 noisy_dir=None,
                 mode='train',
                 sr=16000,
                 frame_len=512,
                 frame_shift=256,
                 block_size=32,
                 block_shift=16,
                 snr_level=[-10, -5, 0, 5, 10, 15, 20]):
        super(MyDataset, self).__init__()
        if os.path.exists(speech_dir):
            NotADirectoryError('Speech dir does not exist')
        if os.path.exists(noise_dir):
            NotADirectoryError('Noise dir does not exist')
        self.speech_dir = speech_dir
        self.noise_dir = noise_dir
        self.mode = mode
        self.noisy_dir = noisy_dir
        self.snr_level = snr_level
        self.sr = sr
        self.frame_len = frame_len
        self.frame_shift = frame_shift
        self.block_size = block_size
        self.block_shift = block_shift
        n = np.log2(self.frame_len)
        if int(n) < n:
            n = int(n) + 1
        self.n_fft = int(2 ** n)
        self.mu = None
        self.sigma = None

        self.speech_wav_files = utils.find_files(self.speech_dir)
        random.shuffle(self.speech_wav_files)
        self.noise_wav_files = utils.find_files(self.noise_dir)
        random.shuffle(self.noise_wav_files)
        if self.mode == 'test':
            if os.path.exists(self.noisy_dir) is False:
                os.mkdir(self.noisy_dir)
            self.noisy_wav_files = utils.find_files(self.noisy_dir)
            self.build_test_pairs()
        else:
            self.data_stats(sample_size=100)  # stats snr information
            self.noise_buffer = np.array([])
            self.speech_buffer = np.array([])
            self.noisy_buffer = np.array([])
            self.wav_idx = 0

    def build_test_pairs(self):
        if len(self.noisy_wav_files) == len(self.speech_wav_files):
            return
        for speech_file in self.speech_wav_files:
            speech_src, _ = sf.read(speech_file)
            noise_file = random.choice(self.noise_wav_files)
            noise_src, _ = sf.read(noise_file)
            while len(noise_src) < len(speech_file):
                noise_file = random.choice(self.noise_wav_files)
                noise_src, _ = sf.read(noise_file)
            snr = random.choice(self.snr_level)
            noise_type = noise_file[noise_file.rfind('/') + 1:noise_file.find('_')]
            noisy_file = self.noisy_dir + os.path.basename(speech_file)[:-4] \
                         + '_' + noise_type + '_' + str(snr) + 'dB.wav'
            speech_len = len(speech_src)
            start_idx = random.randint(0, len(noise_src) - speech_len)
            noise_seg = noise_src[start_idx:start_idx + speech_len]
            noisy_src, _ = utils.add_noise(speech_src, noise_seg, snr)
            sf.write(noisy_file, noisy_src, samplerate=self.sr, subtype='PCM_16')
            self.noisy_wav_files.append(noisy_file)

    def data_stats(self, sample_size):
        """compute sigma and mu of each frequency bin in noise dir, from DeepXi"""
        if os.path.exists(self.noise_dir + 'stats.npz'):
            with np.load(self.noise_dir + 'stats.npz') as stats:
                self.mu = stats['mu_hat']
                self.sigma = stats['sigma_hat']
        else:
            print('Start saving stats')
            samples = []
            for idx in range(sample_size):
                snr = random.choice(self.snr_level)
                speech_file = random.choice(self.speech_wav_files)
                speech_src, _ = librosa.load(speech_file, sr=self.sr)
                noise_file = random.choice(self.noise_wav_files)
                noise_src, _ = librosa.load(noise_file, sr=self.sr)
                start_idx = random.randint(0, len(noise_src) - len(speech_src))
                noise_src = noise_src[start_idx:start_idx + len(speech_src)]
                _, alpha = utils.add_noise(speech_src, noise_src, snr)  # get scale factor based on snr
                noise_src = noise_src * alpha
                # do stft for both speech and noise
                _, sample_speech_mag, _ = utils.analysis(speech_src, self.frame_len, self.frame_shift,
                                                         self.n_fft)
                _, sample_noise_mag, _ = utils.analysis(noise_src, self.frame_len, self.frame_shift,
                                                        self.n_fft)
                # compute prior snr between speech and noise spectrums
                snr_db = utils.prior_snr(sample_speech_mag, sample_noise_mag)  # instantaneous a prior SNR (dB).
                samples.append(np.squeeze(snr_db))
            samples = np.hstack(samples)
            if len(samples.shape) != 2:
                raise ValueError('Incorrect shape for sample.')
            stats = {'mu_hat': np.mean(samples, axis=1), 'sigma_hat': np.std(samples, axis=1)}
            self.mu, self.sigma = stats['mu_hat'], stats['sigma_hat']
            np.savez(self.noise_dir + 'stats.npz', mu_hat=stats['mu_hat'], sigma_hat=stats['sigma_hat'])
            print('Sample statistics saved.')

    def get_train_data(self):
        while len(self.noisy_buffer) < self.block_size * self.frame_len:
            new_speech, _ = librosa.load(self.speech_wav_files[self.wav_idx], sr=self.sr)
            self.speech_buffer = utils.normalize(new_speech)
            while len(self.noise_buffer) < len(self.speech_buffer):
                new_noise, _ = librosa.load(random.choice(self.noise_wav_files), sr=self.sr)
                new_noise = utils.normalize(new_noise)
                self.noise_buffer = np.concatenate((self.noise_buffer, new_noise))
            snr = random.choice(self.snr_level)
            self.noisy_buffer, _ = utils.add_noise(self.speech_buffer, self.noise_buffer[:len(self.speech_buffer)], snr,
                                                   normalization=False)
            self.wav_idx += 1
        speech_block = self.speech_buffer[:self.block_size * self.frame_len]
        noise_block = self.noise_buffer[:self.block_size * self.frame_len]
        noisy_block = self.noisy_buffer[:self.block_size * self.frame_len]
        self.speech_buffer = self.speech_buffer[self.block_shift * self.frame_len:]
        self.noise_buffer = self.noise_buffer[self.block_shift * self.frame_len:]
        self.noisy_buffer = self.noisy_buffer[self.block_shift * self.frame_len:]

        # _, speech_mag, speech_pha = utils.analysis(speech_block, self.frame_len, self.frame_shift, self.n_fft)
        # _, noise_mag, noise_pha = utils.analysis(noise_block, self.frame_len, self.frame_shift, self.n_fft)
        # _, noisy_mag, noisy_pha = utils.analysis(noisy_block, self.frame_len, self.frame_shift, self.n_fft)

        # mapping prior snr to interval[0, 1] using erf function
        # snr_mapped = utils.xi_bar(speech_mag, noisy_mag, self.mu, self.sigma)
        speech_block, noise_block, noisy_block = speech_block[np.newaxis, :], \
                                                 noise_block[np.newaxis, :], noisy_block[np.newaxis, :]  # expand to 3-d
        return speech_block, noise_block, noisy_block

    def get_test_data(self, index):
        noisy_file = self.noisy_wav_files[index]
        basename = os.path.basename(noisy_file)
        speech_num, speaker_name, noise_type, _ = basename.split('_')
        speech_file = self.speech_dir + speech_num + '_' + speaker_name + '.wav'
        noisy_src, _ = librosa.load(noisy_file, sr=self.sr)
        speech_src, _ = librosa.load(speech_file, sr=self.sr)
        speech_len = len(speech_src)
        block_idx = 0
        noisy_blocks, speech_blocks = [], []
        start_idx = 0
        end_idx = start_idx + self.block_size * self.frame_len
        while end_idx < speech_len:
            noisy_block = noisy_src[start_idx: end_idx]
            speech_block = speech_src[start_idx: end_idx]
            noisy_blocks.append(noisy_block[np.newaxis, :])  # expand to 3-d
            speech_blocks.append(speech_block[np.newaxis, :])
            start_idx += self.frame_len * self.block_shift
            end_idx += self.frame_len * self.block_shift
        last_noisy_block = np.zeros_like(noisy_block)
        last_speech_block = np.zeros_like(speech_block)
        last_noisy_block[:len(noisy_src[start_idx:])] = noisy_src[start_idx:]  # padding to same length
        last_speech_block[:len(speech_src[start_idx:])] = speech_src[start_idx:]
        noisy_blocks.append(last_noisy_block[np.newaxis, :])  # expand to 3-d
        speech_blocks.append(last_speech_block[np.newaxis, :])
        return np.array(noisy_blocks), np.array(speech_blocks), basename, speech_len

    def __getitem__(self, index):
        if self.mode == 'train':
            speech_block, noise_block, noisy_block = self.get_train_data()
            return speech_block, noise_block, noisy_block
        elif self.mode == 'test':
            noisy_blocks, speech_blocks, noisy_file, speech_len = self.get_test_data(index)
            return noisy_blocks, speech_blocks, noisy_file, speech_len
        else:
            raise ValueError('No this mode')

    def n_blocks(self):
        cur_blocks = 0
        # print('Start compute num of blocks')
        for speech_wav in self.speech_wav_files:
            wav_info = sf.info(speech_wav)
            n_block = (wav_info.samplerate * wav_info.duration - self.block_size * self.frame_len) // \
                      (self.block_shift * self.frame_len) + 1
            cur_blocks += int(n_block)
        # print('%d speech files,%d blocks' % (len(self.speech_wav_files), cur_blocks))
        return cur_blocks

    def __len__(self):
        """
        num_samples = int(sf.info(self.speech_wav_merged).duration * self.sr)
        n_blocks = num_samples // (self.block_size * self.frame_len)
        return n_blocks
        """
        if self.mode == 'train':
            return self.n_blocks()
        else:
            return len(self.noisy_wav_files)


class TestDataset(Dataset):
    def __init__(self,
                 speech_dir,
                 noisy_dir,
                 sr=16000,
                 pairs=None):
        super(TestDataset, self).__init__()
        self.speech_dir = speech_dir
        self.noisy_dir = noisy_dir
        self.pairs = pairs
        self.sr = sr
        self.speech_wav_files = utils.find_files(self.speech_dir)
        self.noisy_wav_files = utils.find_files(self.noisy_dir)

    def __getitem__(self, index):
        _, speech_file = os.path.split(self.speech_wav_files[index])
        noisy_file = self.noisy_dir + speech_file
        if os.path.exists(noisy_file):
            speech_src, _ = librosa.load(speech_file, sr=self.sr)
            noise_src, _ = librosa.load(noisy_file, sr=self.sr)
            pos_a, pos_b = noisy_file.rfind('_'), noisy_file.rfind('db')
            snr = noisy_file[pos_a + 1:pos_b]
        else:
            raise FileExistsError('No matched noisy file for %s' % speech_file)
        return speech_src, noisy_file, snr

    def __len__(self):
        return len(self.speech_wav_files)


if __name__ == '__main__':
    speech_dir = '/home/dcase/c2019/zx/datasets/clean_testset_wav/'
    noise_dir = '/home/dcase/c2019/zx/datasets/noisex-92-16k/'
    sr = 16000
    frame_len = int(sr * 0.03)  # frame length : fs * 30ms
    frame_shift = int(frame_len // 4)
    snr = [i for i in range(-10, 21, 5)]
    block_size = 33
    dataset = MyDataset(speech_dir, noise_dir, sr, frame_len, frame_shift, block_size, snr)
    dataset.mu
    print(dataset.__len__())
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    for i, x in enumerate(data_loader):
        print(i)
        mag, pha, mask, snr = x
        print(mag.shape)
