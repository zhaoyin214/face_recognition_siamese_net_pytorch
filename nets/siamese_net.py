from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch

from PIL import Image
import numpy as np
import random

import nets.res_net as res_net

class SiameseNet(nn.Module):

    def __init__(self, dim_embedding, is_rgb=False):
        super(SiameseNet, self).__init__()

        # learnable parameters
        self.res_net_50 = res_net.resnet50(num_classes=dim_embedding, is_rgb=is_rgb)


    def forward(self, x1, x2):

        embedding1 = self.res_net_50(x1)
        embedding2 = self.res_net_50(x2)

        return embedding1, embedding2

    def cal_embedding(self, x):
        embedding = self.res_net_50(x)

        return embedding


class ContrastDataset(Dataset):

    def __init__(self, img_folder_dataset,
                 transform=None,
                 keep_order=False):

        self.img_folder_dataset = img_folder_dataset
        self.transform = transform
        self.keep_order = keep_order

    def __getitem__(self, idx):

        if self.keep_order:

            img_path, img_label = self.img_folder_dataset.imgs[idx]
            pil_img = Image.open(img_path)
            # 转为灰度图
            img = pil_img.convert("L")

            if self.transform is not None:
                img = self.transform(img)

            contrastive_label = torch.from_numpy(np.array([0], dtype=np.float32))

            return img, img, contrastive_label, img_path, img_label

        img1_path, img1_label = random.choice(self.img_folder_dataset.imgs)

        # 生成相同id图像对的概率为50%
        is_same_id = random.randint(0, 1)

        if is_same_id:
            while True:
                img2_path, img2_label = random.choice(self.img_folder_dataset.imgs)
                if img1_label == img2_label:
                    break
        else:
            while True:
                img2_path, img2_label = random.choice(self.img_folder_dataset.imgs)
                if img1_label != img2_label:
                    break

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # 转为灰度图
        img1 = img1.convert("L")
        img2 = img2.convert("L")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        contrastive_label = torch.from_numpy(np.array([int(img2_label != img1_label)],
                                                       dtype=np.float32))

        return img1, img2, contrastive_label, img1_label, img2_label

    def __len__(self):
        return len(self.img_folder_dataset)


class ContrastLoss(nn.Module):
    """Contrastive loss function

    Reference:
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2):

        super(ContrastLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):

        euclidean_distance = F.pairwise_distance(x1=embedding1, x2=embedding2, p=2)
        contrast_loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                   label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0), 2))
        return contrast_loss

def cal_distance(embedding1, embedding2):
    euclidean_distance = F.pairwise_distance(x1=embedding1, x2=embedding2, p=2)

    return euclidean_distance


if __name__ == "__main__":

    siamese_net = SiameseNet(10)
    help(siamese_net)

    with open("./siamese_net_structure.txt", "w") as fw:
        print(siamese_net, file=fw)


