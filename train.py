import torch
from torch.utils.data import Dataset, DataLoader
from src.dataloader import *
from loss import zsn2n_loss_func
from src.model.ZSN2N import network
from trainer import train
import gc
from src.device import *

SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 

train_input_files = sorted(list(TRAIN_INPUT_DIR.rglob('*.wav')))
train_target_files = sorted(list(TRAIN_TARGET_DIR.rglob('*.wav')))

test_noisy_files = sorted(list(TEST_NOISY_DIR.rglob('*.wav')))
test_clean_files = sorted(list(TEST_CLEAN_DIR.rglob('*.wav')))

print("No. of Training files:", len(train_input_files))
print("No. of Testing files:", len(test_noisy_files))

# Create datasets
test_dataset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)
train_dataset = SpeechDataset(train_input_files, train_target_files, N_FFT, HOP_LENGTH)

# DEBUG: Check single sample dimensions
print("\n=== INPUT SIZE DEBUGGING ===")
sample_noisy, sample_clean = train_dataset[0]
print(f"Single sample shapes:")
print(f"  Noisy STFT shape: {sample_noisy.shape}")
print(f"  Clean STFT shape: {sample_clean.shape}")

# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# DEBUG: Check batch dimensions
print(f"\nBatch dimensions:")
train_iter = iter(train_loader)
batch_noisy, batch_clean = next(train_iter)
print(f"  Training batch noisy shape: {batch_noisy.shape}")
print(f"  Training batch clean shape: {batch_clean.shape}")

test_iter = iter(test_loader)
test_batch_noisy, test_batch_clean = next(test_iter)
print(f"  Test batch noisy shape: {test_batch_noisy.shape}")
print(f"  Test batch clean shape: {test_batch_clean.shape}")

print(f"\nAudio processing parameters:")
print(f"  Sample Rate: {SAMPLE_RATE}")
print(f"  N_FFT: {N_FFT}")
print(f"  Hop Length: {HOP_LENGTH}")
print(f"  Max Audio Length: {train_dataset.max_len}")
print("=" * 40)

# Rest of your code...
gc.collect()
torch.cuda.empty_cache()

zsn2n = network(N_FFT, HOP_LENGTH).to(DEVICE)
optimizer = torch.optim.Adam(network.parameters())
loss_fn = zsn2n_loss_func
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

train_losses, test_losses = train(network, train_loader, test_loader, loss_fn, optimizer, scheduler, 4)
