import torch
from torch.utils.data import Dataset, DataLoader
from src.dataloader import *
from src.device import *
import gc
from tqdm import tqdm
from loss import getMetricsonLoader, zsn2n_loss_func

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
    
    #print("Actual compute done...testing now")
    
    testmet = getMetricsonLoader(test_loader,net,use_net) # ERROR POINT
    #testmet = None
    # have to integrate this within this test_epoch's function sign 
    # def test(model, noisy_img, clean_img):

    #     with torch.no_grad():
    #         pred = torch.clamp(noisy_img - model(noisy_img),0,1)
    #         MSE = mse(clean_img, pred).item()
    #         PSNR = 10*np.log10(1/MSE)

    #     return PSNR

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_ep_loss, testmet

# keeping original func sign for reference
# def train_epoch(net, train_loader, loss_fn, optimizer):
#     net.train()
#     return train_ep_loss

# this is zsn2n train for one loop
def train_epoch(model, test_dataset, loss_func, optimizer, device="cuda"):
    model.train()
    
    # Get single noisy sample (ZSN2N trains on single samples)
    sample_noisy, _ = test_dataset[0]  # Get first sample, ignore clean
    sample_noisy = sample_noisy.unsqueeze(0).to(device)  # Add batch dim: [1, 2, F, T]
    
    # ZSN2N self-supervised loss
    loss = loss_func(sample_noisy, model)
    
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

        # # ---- initial evaluation only if supervised ----
        # if e == 0 and training_type == "Noise2Clean":
        #     print("Pre-training evaluation")
        #     testmet = getMetricsonLoader(test_loader, net, False)
        #     print("After getmetrics")
        #     with open(basepath + "/results.txt", "w+") as f:
        #         f.write("Initial : \n")
        #         f.write(str(testmet))
        #         f.write("\n")
        
        # # ---- training ----
        # if training_type == "Noise2Noise":
        #     # self-supervised: directly adapt on test set
        #     train_loss = train_epoch(net, test_loader, zsn2n_loss_func, optimizer, test_dataset)
        # else:
        #     # supervised: train on noisy/clean pairs
        #     train_loss = train_epoch(net, train_loader, zsn2n_loss_func, optimizer, test_dataset)
        
        
        # Training (self-supervised on single test sample)
        train_loss = train_epoch(net, test_dataset, zsn2n_loss_func, optimizer)
        
        test_loss = 0
        scheduler.step()
        
        print("Saving model....")
        
        # ---- evaluation ----
        with torch.no_grad():
            test_loss, testmet = test_epoch(net, test_loader, zsn2n_loss_func, use_net=True)

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

# def train(net, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):
    
#     train_losses = []
#     test_losses = []

#     for e in tqdm(range(epochs)):

#         # first evaluating for comparison
        
#         if e == 0 and training_type=="Noise2Clean":
#             print("Pre-training evaluation")
#             #with torch.no_grad():
#             #    test_loss,testmet = test_epoch(net, test_loader, loss_fn,use_net=False)
#             #print("Had to load model.. checking if deets match")
#             testmet = getMetricsonLoader(test_loader,net,False)    # again, modified cuz im loading
#             #test_losses.append(test_loss)
#             #print("Loss before training:{:.6f}".format(test_loss))
        
#             with open(basepath + "/results.txt","w+") as f:
#                 f.write("Initial : \n")
#                 f.write(str(testmet))
#                 f.write("\n")
        
#         # directly giving it test bcs its self supervised
#         train_loss = train_epoch(net, test_loader, loss_fn, optimizer)
#         test_loss = 0
#         scheduler.step()
#         print("Saving model....")
        
#         with torch.no_grad():
#             test_loss, testmet = test_epoch(net, test_loader, loss_fn,use_net=True)

#         train_losses.append(train_loss)
#         test_losses.append(test_loss)
        
#         #print("skipping testing cuz peak autism idk")
        
#         with open(basepath + "/results.txt","a") as f:
#             f.write("Epoch :"+str(e+1) + "\n" + str(testmet))
#             f.write("\n")
        
#         print("OPed to txt")
        
#         torch.save(net.state_dict(), basepath +'/Weights/dc20_model_'+str(e+1)+'.pth')
#         torch.save(optimizer.state_dict(), basepath+'/Weights/dc20_opt_'+str(e+1)+'.pth')
        
#         print("Models saved")

#         # clear cache
#         torch.cuda.empty_cache()
#         gc.collect()

#         #print("Epoch: {}/{}...".format(e+1, epochs),
#         #              "Loss: {:.6f}...".format(train_loss),
#         #              "Test Loss: {:.6f}".format(test_loss))
#     return train_loss, test_loss

