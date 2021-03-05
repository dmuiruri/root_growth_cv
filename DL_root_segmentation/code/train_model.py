import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm

from model import SegRoot
from dataloader import TestDataset, TrainDataset, ValDataset, LoopSampler
from utils import (
    dice_score,
    init_weights,
    evaluate,
    get_ids,
    load_vgg16,
    set_random_seed,
)


# Initialize parameters
dataset_length = 18
seed = 42 # Random seet
width = 8 # width of segRoot
depth = 5 # Depth of Segroot
bs = 2 # Batch size
lr = 0.01 # Learning rate
epochs = 10 # Max number of epochs
verbose = 5 # Interval to save and validate model
num_workers = 2


def train_one_epoch(model, train_iter, optimizer, device):
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    iter = 0
    for x, y in train_iter:
        print("Iteration:", iter)
        iter = iter+1
        print(len(x))
        x, y = x.to(device), y.to(device)
        batch_size = x.shape[0]
        optimizer.zero_grad()
        y_pred = model(x)
        loss = 1 - dice_score(y, y_pred)
        loss = torch.sum(loss) / batch_size
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    # Set random seed
    set_random_seed(seed)

    # Define the device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make train, val, test split
    train_ids, valid_ids, test_ids = get_ids(dataset_length)

    # Load data
    train_data = TrainDataset(train_ids)
    train_iter = DataLoader(train_data, batch_size=bs, num_workers=num_workers, shuffle=True)

    # Load datasets
    train_tdata = TrainDataset(train_ids)
    valid_tdata = ValDataset(valid_ids)
    test_tdata = TestDataset(test_ids)

    # Define model
    model = SegRoot(width, depth).to(device)
    model = model.apply(init_weights)
    

    # define optimizer and lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, verbose=True, patience=5
    )

    # Load pretrained model
    #checkpoint = torch.load("../weights/best_segnet-(8,5)-0.6441.pt", map_location=torch.device(device))
    model.load_state_dict(torch.load("../weights/best_segnet-(8,5)-0.6441.pt"), map_location=torch.device('cpu'))



    print(f"Start training SegRoot-({width},{depth}))......")
    print(f"Random seed is {seed}, batch size is {bs}......")
    print(f"learning rate is {lr}, max epochs is {epochs}......")
    best_valid = float("-inf")
    for epoch in tqdm(range(epochs)):
        train_one_epoch(model, train_iter, optimizer, device)
        if epoch % verbose == 0:
            train_dice = evaluate(model, train_tdata, device)
            valid_dice = evaluate(model, valid_tdata, device)
            scheduler.step(valid_dice)
            print(
                "Epoch {:05d}, train dice: {:.4f}, valid dice: {:.4f}".format(
                    epoch, train_dice, valid_dice
                )
            )
            if valid_dice > best_valid:
                best_valid = valid_dice
                test_dice = evaluate(model, test_tdata, device)
                print("New best validation, test dice: {:.4f}".format(test_dice))
                torch.save(
                    model.state_dict(),
                    f"../weights/best_segnet-({width},{depth}).pt",
                )