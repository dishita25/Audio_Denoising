from pesq import pesq
from scipy import interpolate
import torch
import numpy as np
from src.device import DEVICE
from metrics import AudioMetrics2
import torch.nn.functional as F

# Repeating at a lot of places
SAMPLE_RATE = 24000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000

def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old  = np.linspace(0, duration, original.shape[0])
        time_new  = np.linspace(0, duration, int(original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original


def wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):
    # to time-domain waveform
    y_true_ = torch.squeeze(y_true_, 1)
    # Convert to complex before istft
    y_true_complex = torch.complex(y_true_[..., 0], y_true_[..., 1])
    y_true = torch.istft(y_true_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
    
    x_ = torch.squeeze(x_, 1)
    # Convert to complex before istft
    x_complex = torch.complex(x_[..., 0], x_[..., 1])
    x = torch.istft(x_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)

    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)


    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

wonky_samples = []

def getMetricsonLoader(loader, net, use_net=True):
    net.eval()
    # Original test metrics
    scale_factor = 32768
    # metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI","SNR "]
    metric_names = ["PESQ-WB","PESQ-NB","SNR","SSNR","STOI"]
    overall_metrics = [[] for i in range(5)]
    for i, data in enumerate(loader):
        if (i+1)%10==0:
            end_str = "\n"
        else:
            end_str = ","

        if i in wonky_samples:
            print("Something's up with this sample. Passing...")
        else:
            noisy = data[0]
            clean = data[1]
            if use_net: # Forward of net returns the istft version
                model_output = net(noisy.to(DEVICE))  # This gives you noise prediction in STFT domain

                # For ZSN2N: model predicts noise, so denoised = input - predicted_noise
                denoised_stft = noisy.to(DEVICE) - model_output

                # CORRECT: Handle the channel dimension properly
                denoised_squeezed = torch.squeeze(denoised_stft, 0)  # Remove batch dim: [2, 1537, 215]

                # Split real and imaginary parts from channel dimension
                real_part = denoised_squeezed[0]  # [1537, 215]
                imag_part = denoised_squeezed[1]  # [1537, 215]

                # Create complex tensor with correct shape
                denoised_complex = torch.complex(real_part, imag_part)  # [1537, 215]

                # Now istft should work
                x_est = torch.istft(denoised_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
                x_est_np = x_est.view(-1).detach().cpu().numpy()

                
            else:
                # CORRECT:
                noisy_squeezed = torch.squeeze(noisy, 0)  # Remove batch dim: [2, 1537, 215]
                noisy_real = noisy_squeezed[0]  # [1537, 215]
                noisy_imag = noisy_squeezed[1]  # [1537, 215]
                noisy_complex = torch.complex(noisy_real, noisy_imag)  # [1537, 215]
                x_est_np = torch.istft(noisy_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()


            # CORRECT: Same fix as the model output section
            clean_squeezed = torch.squeeze(clean, 0)  # Remove batch dim: [2, 1537, 215]

            # Split real and imaginary parts from channel dimension  
            clean_real = clean_squeezed[0]  # [1537, 215]
            clean_imag = clean_squeezed[1]  # [1537, 215]

            # Create complex tensor with correct shape
            clean_complex = torch.complex(clean_real, clean_imag)  # [1537, 215]

            x_clean_np = torch.istft(clean_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()


        
            metrics = AudioMetrics2(x_clean_np, x_est_np, 48000)
            
            ref_wb = resample(x_clean_np, 48000, 16000)
            deg_wb = resample(x_est_np, 48000, 16000)
            pesq_wb = pesq(16000, ref_wb, deg_wb, 'wb')
            
            ref_nb = resample(x_clean_np, 48000, 8000)
            deg_nb = resample(x_est_np, 48000, 8000)
            pesq_nb = pesq(8000, ref_nb, deg_nb, 'nb')

            overall_metrics[0].append(pesq_wb)
            overall_metrics[1].append(pesq_nb)
            overall_metrics[2].append(metrics.SNR)
            overall_metrics[3].append(metrics.SSNR)
            overall_metrics[4].append(metrics.STOI)
    print()
    print("Sample metrics computed")
    results = {}
    for i in range(5):
        temp = {}
        temp["Mean"] =  np.mean(overall_metrics[i])
        temp["STD"]  =  np.std(overall_metrics[i])
        temp["Min"]  =  min(overall_metrics[i])
        temp["Max"]  =  max(overall_metrics[i])
        results[metric_names[i]] = temp
    print("Averages computed")
    if use_net:
        addon = "(cleaned by model)"
    else:
        addon = "(pre denoising)"
    print("Metrics on test data",addon)
    for i in range(5):
        print("{} : {:.3f}+/-{:.3f}".format(metric_names[i], np.mean(overall_metrics[i]), np.std(overall_metrics[i])))
    return results


def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)



mse = torch.nn.MSELoss()

def pair_downsampler(img: torch.Tensor):
    """
    Downsamples an STFT image by 2x2 with two complementary filters.
    img: [B, C, H, W] (C can be 2 for real/imag channels)
    returns: two downsampled versions [B, C, H/2, W/2]
    """
    c = img.shape[1]  # number of channels

    # Two 2x2 filters
    filter1 = torch.tensor([[[[0, 0.5],
                              [0.5, 0]]]], dtype=torch.float32, device=img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.tensor([[[[0.5, 0],
                              [0, 0.5]]]], dtype=torch.float32, device=img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    # Apply depthwise conv (groups=c keeps channels separate)
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2


def zsn2n_loss_func(noisy_stft: torch.Tensor, model: torch.nn.Module):
    """
    Zero-Shot Noise2Noise Loss.
    noisy_stft: [B, 2, F, T] - batch with 2 channels (real/imag)
    """
    # Ensure input has correct dimensions
    if len(noisy_stft.shape) == 3:  # [2, F, T]
        noisy_stft = noisy_stft.unsqueeze(0)  # [1, 2, F, T]
    
    # Split into two noisy versions  
    noisy1, noisy2 = pair_downsampler(noisy_stft)
    
    # Predict noise directly
    noise1 = model(noisy1)  
    noise2 = model(noisy2)
    
    # Denoised = noisy - predicted_noise
    pred1 = noisy1 - noise1  
    pred2 = noisy2 - noise2
    
    # Cross-prediction loss (key ZSN2N innovation)
    loss_res = 0.5 * (mse(pred1, noisy2) + mse(pred2, noisy1))
    
    # Consistency loss
    full_noise = model(noisy_stft)
    noisy_denoised = noisy_stft - full_noise
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    
    loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    return loss_res + loss_cons
