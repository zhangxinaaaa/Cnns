# -*- coding: utf-8 -*-
# @Author: xin zhang
# @Date:   2020/7/24 8:39
# @Last Modified by:   xin zhang
# @Last Modified time: 2020/7/24 8:39
import numpy as np
import librosa
import math
import matplotlib.pyplot as plt
import os
import fnmatch
import time



def find_files(dictionary, pattern=['*.wav', '*.WAV'], name_style='root'):
    """
        Find files with specfied patterns in given
    dictionary and its subdictionaries
    Arguments:
        dictionary {[string]} -- [dictionary to find files in]

    Keyword Arguments:
        pattern {list} -- [file extensions] (default: {['*.wav', '*.WAV']})

    Returns:
        [list] -- [contains files' absolute path]
    """
    files = []
    for root, dirnames, filenames in os.walk(dictionary):
        if name_style == 'root':
            for filename in fnmatch.filter(filenames, pattern[0]):
                files.append(os.path.join(root, filename))
            for filename in fnmatch.filter(filenames, pattern[1]):
                files.append(os.path.join(root, filename))
        elif name_style == 'local':
            for filename in fnmatch.filter(filenames, pattern[0]):
                files.append(filename)
            for filename in fnmatch.filter(filenames, pattern[1]):
                files.append(filename)
        else:
            raise ValueError('No this type')
    return files


def merge(src_dir, merged_filename):
    """concatenate all wav files in src dict into one"""
    if src_dir.endswith('/') is False:
        src_dir += '/'
    merged = src_dir + merged_filename
    os.system('cd %s' % src_dir)  # change dir to source dictionary
    src_wav_files = find_files(src_dir)
    tmp = src_dir + 'tmp_500.wav'
    tmp_next = ''
    tmp_to_del = []
    for i in range(0, len(src_wav_files), 500):
        j = i + 500  # merge 500 files once
        filenames = ''
        if j < len(src_wav_files):
            for file in src_wav_files[i:j]:
                filenames += file
                filenames += ' '
        else:
            for file in src_wav_files[i:]:  # last merge
                filenames += file
                filenames += ' '
            os.system('sox %s %s %s' % (filenames, tmp, merged))
            break
        if i == 0:
            os.system('sox %s %s' % (filenames, tmp))
            tmp_to_del.append(tmp)
        else:
            tmp_next = tmp.replace(str(i), str(j))
            while os.path.exists(tmp) is False:
                time.sleep(1)  # waiting for the sox thread
            os.system('sox %s %s %s' % (filenames, tmp, tmp_next))
            tmp_to_del.append(tmp_next)
            tmp = tmp_next
        print('%d files merged' % j)
    for tmp_wav in tmp_to_del:
        os.system('rm %s' % tmp_wav)
    print('Done...Merged file is %s' % merged)


def specgrum(src, frame_len, frame_shift, n_fft, sr=16000, title='Spectrogram'):
    spectrum, freqs, ts, fig = plt.specgram(src, NFFT=n_fft, Fs=sr, window=np.hanning(frame_len),
                                            noverlap=frame_len-frame_shift, mode='default', scale_by_freq=True, sides='default',
                                            scale='dB', xextent=None)  # draw spectrum

    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title(title)
    plt.show()


def filter_matlab(b, a, x):
    y = []
    y.append(b[0] * x[0])
    for i in range(1, len(x)):
        y.append(0)
        for j in range(len(b)):
            if i >= j:
                y[i] = y[i] + b[j] * x[i - j]
                j += 1
        for l in range(len(b) - 1):
            if i > l:
                y[i] = (y[i] - a[l + 1] * y[i - l - 1])
                l += 1
        i += 1
    return y


def normalize(src):
    return src / max(abs(src))


def frames(src, frame_len, frame_shift, sr=16000):
    """get num of frame, return frame num and frame index"""
    n_frames = (len(src) - frame_len) // frame_shift + 1
    times = np.arange(0, n_frames*frame_shift, frame_shift) / sr
    return n_frames, times


def analysis(src, frame_len, frame_shift, n_fft, sr=16000, draw_spec=False):
    """add hanning window and perform stft by overlap and add"""
    src = np.array(src)
    dest = librosa.stft(src, n_fft=n_fft, win_length=frame_len,
                        hop_length=frame_shift, center=False)
    mag = np.abs(dest)
    pha = np.angle(dest)

    if draw_spec:
        specgrum(src, frame_len, frame_shift, n_fft, sr)
    return dest, mag, pha


def synthesis(src, frame_len, frame_shift, draw_wave=False):
    src = np.array(src)
    dest = librosa.istft(src, win_length=frame_len, hop_length=frame_shift, center=False)

    return dest


def pre_emphasis(src, b=[1, 0.95], a=[1]):
    return filter_matlab(b, a, src)


def de_emphasis(src, b=[1], a=[1, 0.95]):
    return filter_matlab(b, a, src)


def add_noise(clean, noise, snr):
    Pc = np.dot(clean.T, clean) / len(clean)
    Pn = np.dot(noise.T, noise) / len(noise)
    alpha = np.sqrt(Pc / Pn / (10 ** (snr / 10)))  # scaling factor
    return alpha * noise + clean, alpha


def prior_snr(mag_clean, mag_noise):
    """Instantaneous a priori SNR in dB between speech and noise spectrums."""
    P_c = mag_clean ** 2
    P_n = mag_noise ** 2
    snr_db = 10 * np.log10(np.maximum(P_c / np.maximum(P_n, 1e-12), 1e-12))

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
    snr_mapped = np.zeros_like(snr_db)

    for i in range(snr_db.shape[0]):  # frequency bin
        for j in range(snr_db.shape[1]):  # time index
            snr_mapped[i, j] = \
                0.5 * (1 + math.erf((snr_db[i, j] - mu[i]) / sigma[i] * math.sqrt(2.0)))

    return snr_mapped


if __name__ == "__main__":
    src_dir = '/home/dcase/c2019/zx/se/deepxi/set/train_clean_speech/'
    merge(src_dir, 'xiaozhu.wav')
