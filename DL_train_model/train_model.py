'''
Author: Mikko Saukkoriipi
Date: 20 April 2021
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import random
from tqdm import tqdm
import pickle
from pathlib import Path

from model import SegNet
from dataloader import TestDataset, TrainDataset, ValDataset

# Initialize parameters
bs = 2 # Batch size
lr = 0.01 # Learning rate
epochs = 200 # Max number of epochs
verbose = 2 # Interval to validate model and save if it is better than last best
num_workers = 10 # Number of workers

# Define paths
images_dir = Path('data/train_images_and_masks')
output_weights_dir = "trained_segnet_weights.pt"
model_dir = "pretrained_segnet.pth"

# Train-val-test split. 60-20-20.
def get_ids(length_dataset):
    ids = list(range(length_dataset))
    random.shuffle(ids)
    train_split = round(0.6 * length_dataset)
    t_v_spplit = (length_dataset - train_split) // 2
    train_ids = ids[:train_split]
    valid_ids = ids[train_split:train_split+t_v_spplit]
    test_ids = ids[train_split+t_v_spplit:]
    return train_ids, valid_ids, test_ids

def dice_score(y, y_pred, smooth=1.0, thres=0.9):
    n = y.shape[0]
    y = y.view(n, -1)
    y_pred = y_pred.view(n, -1)
    num = 2 * torch.sum(y * y_pred, dim=1, keepdim=True) + smooth
    den = torch.sum(y, dim=1, keepdim=True) + \
        torch.sum(y_pred, dim=1, keepdim=True) + smooth
    score = num / den
    return score

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)

def evaluate(model, dataset, device, thres=0.9):
    model.eval()
    torch.cuda.empty_cache()    
    num, den = 0, 0
    # shutdown the autograd
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            y_pred = model(x)
            y = y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
            y_pred[y_pred>=thres] = 1.0
            y_pred[y_pred<thres] = 0.0
            num += 2 * (y_pred * y).sum()
            den += y_pred.sum() + y.sum()
    torch.cuda.empty_cache() 
    return num / den

def train_one_epoch(model, train_iter, optimizer, device):
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    for x, y in train_iter:
        x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
        optimizer.zero_grad()
        y_pred = model(x)
        loss = 1 - dice_score(y, y_pred)
        loss = torch.sum(loss) / batch_size
        loss.backward()
        optimizer.step()

def create_images_index_pikle_file(images_dir):

    # Read paths in images_dir folder
    imgs = sorted(list(images_dir.glob('*.jpg')))

    # Get paths
    im_list = [im.parts[-1] for im in imgs]

    # Drop images which end with "-mask.jpg"
    for im in im_list:
        if im[-9:] == "-mask.jpg":
            im_list.remove(im)

    # Get dataset lenght
    dataset_length = len(im_list)
    print('Number of images : ', dataset_length)

    # Create paths dict
    info_dict = {k: v for k, v in enumerate(im_list)}

    # Save idex info in a dictionary
    with open('data/info.pkl', 'wb') as handle:
        pickle.dump(info_dict, handle)
    print('Images index info and paths saved to data/info.pkl ! \n')

    # Return number of images
    return dataset_length

if __name__ == "__main__":

    # Create images index file. Used in data reading and to make train-val-test split.
    dataset_length = create_images_index_pikle_file(images_dir)

    # Define the device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make train, val, test split
    train_ids, valid_ids, test_ids = get_ids(dataset_length)

    # Load datasets
    valid_tdata = ValDataset(valid_ids)
    test_tdata = TestDataset(test_ids)
    train_tdata = TrainDataset(train_ids)
    train_iter = DataLoader(train_tdata, batch_size=bs, num_workers=num_workers, shuffle=True)

    # Define model
    model = SegNet(8, 5).to(device)

    # Add initial weight to model
    model = model.apply(init_weights)

    # Define optimizer and lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, verbose=True, patience=5
    )

    # Load pretrained model "transfer learning"
    if device.type == "cpu":
        print("No Cuda available, load pretrained segroot weights to CPU")
        checkpoint = torch.load(model_dir, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Load pretrained weights to Cuda GPU")
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Print training parameters before starting training
    print(f"\nStart training SegNet model......")
    print(f"Batch size is {bs}, learning rate is {lr}, max epochs is {epochs}......")

    # Get starting results from pretrained model
    best_valid = evaluate(model, valid_tdata, device)
    print("\nStarting train set dice score: {:.4f}".format(evaluate(model, train_tdata, device)))
    print("Starting validation set dice score: {:.4f}\n".format(best_valid))

    # Train the model
    for epoch in tqdm(range(epochs)):
        train_one_epoch(model, train_iter, optimizer, device)

        # Check results
        if epoch % verbose == 0:
            train_dice = evaluate(model, train_tdata, device)
            valid_dice = evaluate(model, valid_tdata, device)
            scheduler.step(valid_dice)
            print("Epoch {:05d}, train dice: {:.4f}, valid dice: {:.4f}".format(epoch, train_dice, valid_dice))

            # Save model if validation dice score is better than before
            if valid_dice > best_valid:
                best_valid = valid_dice

                # Get and print test set dice score
                test_dice = evaluate(model, test_tdata, device)
                print("New best result, test dice: {:.4f}".format(test_dice))

                # Save weights used in prediction
                torch.save(model.state_dict(), output_weights_dir)

                # Save weights and optimizer used in further training
                state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, model_dir)
