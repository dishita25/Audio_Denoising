import torch
import torch.nn.functional as F
import torch.nn as nn

SAMPLE_RATE = 24000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 

class network(nn.Module):
    def __init__(self, n_chan, chan_embed=48, n_fft=N_FFT, hop_length=HOP_LENGTH):
        super(network, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)

        self.conv4 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.conv4(x)
        return x
