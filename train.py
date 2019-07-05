#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from configs.config import config
from nets.siamese_net import SiameseNet, ContrastDataset, ContrastLoss
from utils.utils import plot_history, imshow

import pickle
import time
import copy
from tqdm import tqdm
import sys
import os


# train
# %%
def train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs=25,
                early_stopping_patience=None,
                reduce_lr_on_plateau=None):

    history = dict(epoch=[],
                   train_loss=[],
                   val_loss=[])

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = sys.float_info.max

    if early_stopping_patience is not None:
        early_stopping_cnt = 0

    if reduce_lr_on_plateau is not None:
        reduce_lr_on_plateau_cnt = 0

    for epoch in range(num_epochs):
        print("-" * 10)
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0

            # progress bar
            pbar = tqdm(total=len(data_loaders[phase]),
                        desc=phase,
                        ascii=True)

            # Iterate over data.
            for data in data_loaders[phase]:
                img1s = data[0].to(device)
                img2s = data[1].to(device)
                labels = data[2].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    embedding1s, embedding2s = model(img1s, img2s)
                    loss = criterion(embedding1s, embedding2s, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * labels.size(0)
                pbar.update(1)

            epoch_loss = running_loss / dataset_size[phase]
            pbar.close()

            print("{} Loss: {:.4f}".format(
                phase, epoch_loss))

            # history
            if phase == "train":
                history["epoch"].append(epoch)
                history["train_loss"].append(epoch_loss)
            elif phase == "val":
                history["val_loss"].append(epoch_loss)
            else:
                pass

            # early stopping
            if early_stopping_patience is not None:
                if phase == "val" and epoch_loss >= best_loss:
                    early_stopping_cnt += 1
                elif phase == "val" and epoch_loss < best_loss:
                    early_stopping_cnt = 0
                else:
                    pass

                if early_stopping_cnt >= early_stopping_patience:
                    print("Early Stopping...")
                    # load best model weights
                    model.load_state_dict(best_model_wts)
                    return model, history

            # reduce lr on plateau
            if reduce_lr_on_plateau is not None:
                if phase == "val" and epoch_loss >= best_loss:
                    reduce_lr_on_plateau_cnt += 1
                elif phase == "val" and epoch_loss < best_loss:
                    reduce_lr_on_plateau_cnt = 0
                else:
                    pass

                if reduce_lr_on_plateau_cnt >= reduce_lr_on_plateau["patience"]:
                    reduce_lr_on_plateau_cnt = 0
                    print("Error Plateau, Reducing the Learning Rate...")
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= reduce_lr_on_plateau["factor"]
                    print("Learning Rate: {}".format(param_group["lr"]))

            # best save according to val_loss
            if phase == "val" and epoch_loss < best_loss:
                print("Best Save...")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),
                           "./output/best_model-epoch_{}-val_loss_{:.4f}.pth".format(
                               epoch, epoch_loss))
                print("./output/best_model-epoch_{}-val_loss_{:.4f}.pth".format(
                    epoch, epoch_loss))

        print("\n\n")

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


# main
#%%
if __name__ == "__main__":

    # dataset
    image_datasets = {x: ImageFolder(root=config[x + "_set_root"],
                                     transform=None)
                    for x in ["train", "val"]}

    dataset_size = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    print("number of train identities: {}".format(len(image_datasets["train"].classes)))
    print("number of train faces: {}".format(dataset_size["train"]))
    print("number of val identities: {}".format(dataset_size["val"]))
    print("number of val faces: {}".format(len(image_datasets["val"].classes)))

    # data augmentation
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomRotation(10),
            transforms.Resize(100),
            transforms.RandomResizedCrop(90),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(100),
            transforms.CenterCrop(90),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # contrast dataset
    contrast_datasets = {x: ContrastDataset(img_folder_dataset=image_datasets[x],
                                            transform=data_transforms[x])
                        for x in ["train", "val"]}

    # visualization
    samples_loader =  DataLoader(dataset=contrast_datasets["train"],
                                 batch_size=8,
                                 shuffle=False,
                                 num_workers=0)
    batch_samples = iter(samples_loader).next()
    batch_imgs = torch.cat((batch_samples[0], batch_samples[1]), 0)
    torch_img = make_grid(batch_imgs)
    imshow(torch_img, is_torch=True)
    print(batch_samples[2].numpy())

    # data loader
    data_loaders = {x: torch.utils.data.DataLoader(dataset=contrast_datasets[x],
                                                   batch_size=config[x + "_batch_size"],
                                                   shuffle=True,
                                                   num_workers=0)
                    for x in ["train", "val"]}
    dataset_size = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # network
    net = SiameseNet(dim_embedding=config["dim_embedding"],
                     is_rgb=config["is_rgb"])
    if os.path.isfile("./output/best_model.pth"):
        net.load_state_dict(torch.load("./output/best_model.pth"))
    net.to(device)

    # loss
    criterion = ContrastLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Decay LR by a factor of 0.5 every 2 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    net, history = train_model(model=net,
                               data_loaders=data_loaders,
                               criterion=criterion,
                               optimizer=optimizer,
                               scheduler=exp_lr_scheduler,
                               num_epochs=config["train_epochs"],
                               early_stopping_patience=config["early_stopping_patience"],
                               reduce_lr_on_plateau = config["reduce_lr_on_plateau"])

    with open("./output/history.pickle", "wb") as fw:
        pickle.dump(history, fw)

    plot_history(history, "./output/history.png")
