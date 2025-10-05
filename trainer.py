import torch
from torch.utils.data import Dataset, DataLoader
from src.dataloader import *
from src.device import *
import gc
from tqdm import tqdm
from loss import getMetricsonLoader, zsn2n_loss_func

import matplotlib.pyplot as plt
import numpy as np

# Repeating at a lot of places
SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 

train_input_files = sorted(list(TRAIN_INPUT_DIR.rglob('*.wav')))
train_target_files = sorted(list(TRAIN_TARGET_DIR.rglob('*.wav')))

test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))
test_clean_files = sorted(list(TEST_CLEAN_DIR.rglob('*.wav')))

print("No. of Training files:",len(train_input_files))
print("No. of Testing files:",len(test_noisy_files))

test_dataset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)
train_dataset = SpeechDataset(train_input_files, train_target_files, N_FFT, HOP_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)

# For testing purpose
test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)


def test_epoch(net, test_loader, loss_fn, use_net=True):
    net.eval()
    test_ep_loss = 0.
    counter = 0.
    '''
    for noisy_x, clean_x in test_loader:
        # get the output from the model
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        # Calc the metrics here
        test_ep_loss += loss.item() 
        
        counter += 1

    test_ep_loss /= counter
    '''
       
    testmet = getMetricsonLoader(test_loader,net,use_net) # ERROR POINT

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_ep_loss, testmet



# this is zsn2n train for one loop
def train_epoch(model, test_dataset, loss_func, optimizer, device="cuda"):
    model.train()
    
    # Get single noisy sample (ZSN2N trains on single samples)
    sample_noisy, _ = test_dataset[0]  # Get first sample, ignore clean
    sample_noisy = sample_noisy.unsqueeze(0).to(device)  # Add batch dim: [1, 2, F, T]
    
    # ZSN2N self-supervised loss
    loss = loss_func(sample_noisy, model)
    print("********Losses**********")
    print(loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



def denoise(model, noisy_img):

    with torch.no_grad():
        pred = torch.clamp( noisy_img - model(noisy_img),0,1)

    return pred



def train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs, test_dataset):
    
    train_losses = []
    test_losses = []

    for e in tqdm(range(epochs)):

        # Training (self-supervised on single test sample)
        train_loss = train_epoch(net, test_dataset, zsn2n_loss_func, optimizer)
        
        test_loss = 0
        scheduler.step()
        
        print("Saving model....")
        
        # ---- evaluation ----
        with torch.no_grad():
            test_loss, testmet = test_epoch(net, test_loader, zsn2n_loss_func, use_net=True)


        if e == 0 or (e + 1) % 5 == 0:  # Show plots at epoch 1 and every 5th epoch
            print(f"=== EPOCH {e+1}: SHOWING MODEL DENOISING RESULTS ===")
            
            # Get a test sample
            net.eval()
            with torch.no_grad():
                for noisy_x, clean_x in test_loader:
                    noisy_x = noisy_x.to(DEVICE)
                    clean_x = clean_x.to(DEVICE)
                    
                    # Get model prediction - ZSN2N predicts NOISE
                    predicted_noise = net(noisy_x)
                    
                    # Get denoised signal by subtracting predicted noise
                    denoised_x = noisy_x - predicted_noise
                    
                    # Convert STFT to waveform for plotting
                    def stft_to_wav(stft_tensor):
                        if len(stft_tensor.shape) == 4:
                            stft_tensor = stft_tensor.squeeze(0)  # Remove batch dim
                        real = stft_tensor[0].cpu()
                        imag = stft_tensor[1].cpu()
                        complex_stft = torch.complex(real, imag)
                        wav = torch.istft(complex_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
                        return wav.numpy()
                    
                    # Convert all to waveforms
                    noisy_wav = stft_to_wav(noisy_x)
                    denoised_wav = stft_to_wav(denoised_x) 
                    clean_wav = stft_to_wav(clean_x)
                    
                    # Create time axis
                    time = np.arange(len(noisy_wav)) / SAMPLE_RATE
                    
                    # Plot the results
                    plt.figure(figsize=(15, 10))
                    
                    plt.subplot(3, 1, 1)
                    plt.plot(time, noisy_wav, 'r-', alpha=0.7, linewidth=0.5)
                    plt.title(f'EPOCH {e+1}: Input Noisy Signal')
                    plt.ylabel('Amplitude')
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(3, 1, 2)
                    plt.plot(time, denoised_wav, 'b-', alpha=0.7, linewidth=0.5)
                    plt.title(f'EPOCH {e+1}: Model Denoised Output')
                    plt.ylabel('Amplitude')
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(3, 1, 3)
                    plt.plot(time, clean_wav, 'g-', alpha=0.7, linewidth=0.5)
                    plt.title(f'EPOCH {e+1}: Target Clean Signal')
                    plt.ylabel('Amplitude')
                    plt.xlabel('Time (seconds)')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(f'/kaggle/working/epoch_{e+1}_denoising_result.png', dpi=150)
                    plt.show()
                    
                    print(f"Denoising result saved as epoch_{e+1}_denoising_result.png")
                    break


        train_losses.append(train_loss)
        test_losses.append(test_loss)

        #testmet = "hi"
        
        with open(basepath + "/results.txt", "a") as f:
            f.write("Epoch :" + str(e+1) + "\n" + str(testmet))
            f.write("\n")
        
        print("Results written")
        
        torch.save(net.state_dict(), basepath + '/Weights/dc20_model_' + str(e+1) + '.pth')
        torch.save(optimizer.state_dict(), basepath + '/Weights/dc20_opt_' + str(e+1) + '.pth')
        
        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()
    
    return train_loss, test_loss
