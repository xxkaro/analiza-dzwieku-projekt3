import numpy as np
import scipy.fftpack

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def framing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    frame_len = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_len = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_len - frame_len)) / frame_step)) + 1

    pad_signal_len = num_frames * frame_step + frame_len
    z = np.zeros((pad_signal_len - signal_len))
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def hamming_window(frames):
    return frames * np.hamming(frames.shape[1])

def compute_fft(frames, NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    return pow_frames

def mel_filterbank(pow_frames, sample_rate, nfilt=26, NFFT=512):
    low_freq = 0
    high_freq = sample_rate / 2
    mel_points = np.linspace(hz_to_mel(low_freq), hz_to_mel(high_freq), nfilt + 2)
    hz_points = mel_to_hz(mel_points)
    bin = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = bin[m - 1]
        f_m = bin[m]
        f_m_plus = bin[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    return np.log(filter_banks)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)

def dct_filterbanks(filter_banks, num_ceps=13):
    M = filter_banks.shape[1]  
    N = filter_banks.shape[0] 
    mfccs = np.zeros((N, num_ceps))

    for n in range(N):  
        log_energy = np.log(np.sum(filter_banks[n]))  
        for k in range(1, num_ceps):  
            sum_val = 0
            for m in range(M):
                sum_val += filter_banks[n, m] * np.cos(np.pi * k * (m - 0.5) / M)
            coeff = np.sqrt(2 / M)
            mfccs[n, k] = coeff * sum_val

        mfccs[n, 0] = log_energy
    
    # mfccs = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]

    return mfccs

def extract_mfcc(signal, sample_rate):
    emphasized = pre_emphasis(signal)
    frames = framing(emphasized, sample_rate)
    frames = hamming_window(frames)
    pow_frames = compute_fft(frames)
    mel_banks = mel_filterbank(pow_frames, sample_rate)
    mfcc = dct_filterbanks(mel_banks)
    return np.mean(mfcc, axis=0)

