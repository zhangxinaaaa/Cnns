import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import fnmatch
import soundfile as sf
import librosa
import utils
import random


def stream_length(wav_files):
    total_len = 0
    for wav_file in wav_files:
        wav_info = sf.info(wav_file)
        total_len = total_len + wav_info.duration
    return total_len


class MyDataset(Dataset):
    def __init__(self,
                 speech_dir,
                 noise_dir,
                 sr,
                 frame_len,
                 frame_shift,
                 block_size,
                 snr_level=[-10, -5, 0, 5, 10, 15, 20]):
        if os.path.exists(speech_dir):
            NotADirectoryError('Speech dir does not exist')
        if os.path.exists(noise_dir):
            NotADirectoryError('Noise dir does not exist')
        self.speech_dir = speech_dir
        self.noise_dir = noise_dir
        self.snr_level = snr_level
        self.sr = sr
        self.frame_len = frame_len
        self.frame_shift = frame_shift
        self.block_size = block_size
        n = np.log2(self.frame_len)
        if int(n) < n:
            n = int(n) + 1
        self.n_fft = int(2 ** n)
        self.mu = None
        self.sigma = None
        self.speech_stream = None
        self.noise_stream = None
        self.speech_wav_files = utils.find_files(self.speech_dir)
        self.speech_wav_merged = self.speech_dir + 'speech_merged.wav'
        self.noise_wav_merged = self.noise_dir + 'noise_merged.wav'
        if os.path.exists(self.speech_wav_merged) is False:
            utils.merge(self.speech_dir, self.speech_wav_merged)
        if os.path.exists(self.noise_wav_merged) is False:
            utils.merge(self.noise_dir, self.noise_wav_merged)
        self.data_stats(sample_size=1000)  # statistical noise information
        self.set_stream()
        self.speech_block_buffer = np.empty(1)
        self.wav_idx = -1

    def data_stats(self, sample_size):
        """compute sigma and mu of frequency bin in noise dir, from DeepXi"""
        if os.path.exists(self.noise_dir + 'stats.npz'):
            with np.load(self.noise_dir + 'stats.npz') as stats:
                self.mu = stats['mu_hat']
                self.sigma = stats['sigma_hat']
        else:
            print('Start saving stats')
            self.set_stream()
            samples = []
            for idx in range(sample_size):
                snr = random.choice(self.snr_level)
                sample_speech_src = self.speech_stream.__next__()
                sample_noise_src = self.noise_stream.__next__()
                _, alpha = utils.add_noise(sample_speech_src, sample_noise_src, snr)  # get scale factor based on snr
                sample_noise_src = sample_noise_src * alpha
                _, sample_speech_mag, _ = utils.analysis(sample_speech_src, self.frame_len, self.frame_shift,
                                                         self.n_fft)
                _, sample_noise_mag, _ = utils.analysis(sample_noise_src, self.frame_len, self.frame_shift,
                                                        self.n_fft)
                snr_db = utils.prior_snr(sample_speech_mag, sample_noise_mag)  # instantaneous a prior SNR (dB).
                samples.append(np.squeeze(snr_db))
            samples = np.hstack(samples)
            if len(samples.shape) != 2:
                raise ValueError('Incorrect shape for sample.')
            stats = {'mu_hat': np.mean(samples, axis=1), 'sigma_hat': np.std(samples, axis=1)}
            self.mu, self.sigma = stats['mu_hat'], stats['sigma_hat']
            np.savez(self.noise_dir + 'stats.npz', mu_hat=stats['mu_hat'], sigma_hat=stats['sigma_hat'])
            print('Sample statistics saved.')

    def set_stream(self, stream_type='both'):
        if stream_type == 'both':
            self.speech_stream = librosa.stream(self.speech_wav_merged,
                                                block_length=self.block_size,
                                                frame_length=self.frame_len,
                                                hop_length=self.frame_len)
            self.noise_stream = librosa.stream(self.noise_wav_merged,
                                               block_length=self.block_size,
                                               frame_length=self.frame_len,
                                               hop_length=self.frame_len)
        elif stream_type == 'speech':
            self.speech_stream = librosa.stream(self.speech_wav_merged,
                                                block_length=self.block_size,
                                                frame_length=self.frame_len,
                                                hop_length=self.frame_len)
        elif stream_type == 'noise':
            self.noise_stream = librosa.stream(self.noise_wav_merged,
                                               block_length=self.block_size,
                                               frame_length=self.frame_len,
                                               hop_length=self.frame_len)
        else:
            raise ValueError('No this stream mode!')

    def __getitem__(self, index):
        if index < self.__len__():
            snr = random.choice(self.snr_level)
            while len(self.speech_block_buffer) < self.block_size * self.frame_len:
                self.speech_block_buffer = librosa.load(self.speech_wav_files[self.wav_idx], sr=self.sr)
                self.speech_block_buffer = utils.normalize(self.speech_block_buffer)
                self.wav_idx += 1
            speech_block = self.speech_block_buffer[:self.block_size * self.frame_len]
            if len(self.speech_block_buffer) > self.block_shift:
                self.speech_block_buffer = self.speech_block_buffer[self.block_shift:]
            try:
                noise_block = self.noise_stream.__next__()
                if len(noise_block) != self.block_size * self.frame_len:
                    raise StopIteration
            except StopIteration:
                self.set_stream(stream_type='noise')
                noise_block = self.noise_stream.__next__()
            noisy_block, _ = utils.add_noise(speech_block, noise_block, snr)

            _, speech_mag, speech_pha = utils.analysis(speech_block, self.frame_len, self.frame_shift, self.n_fft)
            _, noise_mag, noise_pha = utils.analysis(noise_block, self.frame_len, self.frame_shift, self.n_fft)
            _, noisy_mag, noisy_pha = utils.analysis(noisy_block, self.frame_len, self.frame_shift, self.n_fft)

            # mapping prior snr to interval[0, 1] using erf function
            snr_mapped = utils.xi_bar(speech_mag, noisy_mag, self.mu, self.sigma)
            speech_block, noisy_block = speech_block[np.newaxis, :], noisy_block[np.newaxis, :]
            return noisy_mag, noisy_pha, speech_block, noisy_block, snr_mapped

    def n_blocks(self):
        cur_blocks = 0
        for speech_wav in self.speech_wav_files:
            wav_info = sf.info(speech_wav)
            n_block = (wav_info.samplerate * wav_info.duration - self.block_size*self.frame_len) // self.block_shift
            cur_blocks += n_block
        return cur_blocks

    def __len__(self):
        """
        num_samples = int(sf.info(self.speech_wav_merged).duration * self.sr)
        n_blocks = num_samples // (self.block_size * self.frame_len)
        return n_blocks
        """
        return self.n_blocks()



if __name__ == '__main__':
    speech_dir = '/home/dcase/c2019/zx/datasets/clean_testset_wav/'
    noise_dir = '/home/dcase/c2019/zx/datasets/noisex-92-16k/'
    sr = 16000
    frame_len = int(sr * 0.03)  # frame length : fs * 30ms
    frame_shift = int(frame_len // 4)
    snr = [i for i in range(-10, 21, 5)]
    block_size = 33
    dataset = MyDataset(speech_dir, noise_dir, sr, frame_len, frame_shift, block_size, snr)
    print(dataset.__len__())
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    for i, x in enumerate(data_loader):
        print(i)
        mag, pha, mask, snr = x
        print(mag.shape)
