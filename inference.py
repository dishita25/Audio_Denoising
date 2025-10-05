import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from src.model.ZSN2N import network
from src.dataloader import SpeechDataset
from src.device import DEVICE
from metrics import AudioMetrics

# Repeating at a lot of places
SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000

model_weights_path = "/kaggle/working/Audio_Denoising/white_Noise2Noise/Weights/dc20_model_5.pth"

zsn2n = network(N_FFT, HOP_LENGTH).to(DEVICE)
optimizer = torch.optim.Adam(zsn2n.parameters())

checkpoint = torch.load(model_weights_path,
                        map_location=torch.device('cpu')
                       )


test_noisy_files = sorted(list(Path("Samples/Sample_Test_Input").rglob('*.wav')))
test_clean_files = sorted(list(Path("Samples/Sample_Test_Target").rglob('*.wav')))

test_dataset = SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)

# For testing purpose
test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)

zsn2n.load_state_dict(checkpoint)

index = 4

zsn2n.eval()
test_loader_single_unshuffled_iter = iter(test_loader_single_unshuffled)

x_n, x_c = next(test_loader_single_unshuffled_iter)
for _ in range(index):
    x_n, x_c = next(test_loader_single_unshuffled_iter)

x_est = zsn2n(x_n, is_istft=True)

x_est_np = x_est[0].view(-1).detach().cpu().numpy()
x_c_np = torch.istft(torch.squeeze(x_c[0], 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()
x_n_np = torch.istft(torch.squeeze(x_n[0], 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()

metrics = AudioMetrics(x_c_np, x_est_np, SAMPLE_RATE)
print(metrics.display())