import torch
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from config import config
from nets.siamese_net import SiameseNet, ContrastDataset, ContrastLoss
from utils.utils import plot_loss, imshow

train_set = ImageFolder(root=config["train_set_root"])
transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.RandomRotation(10),
                                transforms.RandomCrop((90, 90)),
                                transforms.ToTensor()])

contrast_set = ContrastDataset(img_folder_dataset=train_set,
                               transform=transform)
len_contrast_set = len(contrast_set)

# visualization
samples_loader =  DataLoader(dataset=contrast_set,
                             batch_size=8,
                             shuffle=False,
                             num_workers=0)
batch_samples = iter(samples_loader).next()
batch_imgs = torch.cat((batch_samples[0], batch_samples[1]), 0)
torch_img = make_grid(batch_imgs)
imshow(torch_img, is_torch=True)
print(batch_samples[2].numpy())

# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# network
net = SiameseNet(dim_embedding=config["dim_embedding"],
                 is_rgb=config["is_rgb"])
net.to(device)

# loss
criterion = ContrastLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

# train
train_set_loader = DataLoader(dataset=contrast_set,
                              batch_size=config["train_batch_size"],
                              shuffle=True,
                              num_workers=0)

counter = []
loss_history = []

for epoch in range(config["train_epochs"]):

    for idx, data in enumerate(train_set_loader):

        img1s, img2s, labels = data[0].to(device), data[1].to(device), data[2].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        embedding1s, embedding2s = net(img1s, img2s)
        train_loss = criterion(embedding1s, embedding2s, labels)
        train_loss.backward()
        optimizer.step()

        if idx % 10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch, train_loss.item()))
            counter.append(idx + epoch * len_contrast_set)
            loss_history.append(train_loss.item())

plot_loss(counter, loss_history)

# 保存模型
torch.save(net.state_dict(), "best.siamese.ph")
