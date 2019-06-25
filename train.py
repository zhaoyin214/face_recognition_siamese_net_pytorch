import torch
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from config import config
from siamese_net import SiameseNet, ContrastDataset, ContrastLoss
from utils import plot_loss, imshow

training_set = ImageFolder(root=config["training_set_dir"])
transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.RandomRotation(10),
                                transforms.RandomCrop((90, 90)),
                                transforms.ToTensor()])

contrast_set = ContrastDataset(img_folder_dataset=training_set,
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
net.to(device)

# network
net = SiameseNet(dim_embedding=config["dim_embedding"],
                 is_rgb=config["is_rgb"])

# loss
criterion = ContrastLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

# training
training_set_loader = DataLoader(dataset=contrast_set,
                                 batch_size=config["training_batch_size"],
                                 shuffle=True,
                                 num_workers=0)

counter = []
loss_history = []

for epoch in range(config["training_epochs"]):

    for idx, data in enumerate(training_set_loader):

        img1s, img2s, labels = data[0].to(device), data[1].to(device), data[2].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        embedding1s, embedding2s = net(img1s, img2s)
        training_loss = criterion(embedding1s, embedding2s, labels)
        training_loss.backward()
        optimizer.step()

        if idx % 10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch, training_loss.item()))
            counter.append(idx + epoch * len_contrast_set)
            loss_history.append(training_loss.item())

plot_loss(counter, loss_history)

# 保存模型
torch.save(net.state_dict(), "best.siamese.ph")
