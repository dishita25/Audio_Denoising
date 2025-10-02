import torch
import torch.nn.functional as F
import torch.nn as nn

SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 

# class network(nn.Module):
#     def __init__(self, n_chan, chan_embed=48, n_fft=N_FFT, hop_length=HOP_LENGTH):
#         super(network, self).__init__()

#         self.n_fft = n_fft
#         self.hop_length = hop_length

#         self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#         self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
#         self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)

#         self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.act(self.conv2(x))
#         x = self.conv3(x)
#         return x

class network(nn.Module):
    def __init__(self, n_chan, chan_embed=48, n_fft=N_FFT, hop_length=HOP_LENGTH, is_istft=True):
        super(network, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x, is_istft):
        # Original network processing - this estimates the noise
        noise_estimate = self.act(self.conv1(x))
        noise_estimate = self.act(self.conv2(noise_estimate))
        noise_estimate = self.conv3(noise_estimate)
        
        # ZSN2N approach: subtract estimated noise from input
        denoised_stft = x - noise_estimate
        
        if is_istft:
            # Convert STFT back to time-domain audio
            return self._stft_to_audio(denoised_stft)
        else:
            # Return STFT domain output
            return denoised_stft
    
    def _stft_to_audio(self, stft_tensor):
        """Convert STFT tensor back to audio waveform"""
        # stft_tensor shape: [batch, 2, freq, time]
        
        # Extract real and imaginary parts
        real = stft_tensor[:, 0, :, :]  # [batch, freq, time]
        imag = stft_tensor[:, 1, :, :]  # [batch, freq, time]
        
        # Create complex tensor
        complex_stft = torch.complex(real, imag)
        
        # Convert to audio using ISTFT
        audio = torch.istft(complex_stft, 
                           n_fft=self.n_fft, 
                           hop_length=self.hop_length, 
                           normalized=True)
        
        return audio
