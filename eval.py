import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import pandas as pd
import os

from siamese_net import SiameseNet, ContrastDataset, cal_distance
from config import config
from utils import imshow

# test
test_set = ImageFolder(root=config["test_set_dir"])
transform = transforms.Compose([transforms.Resize((90, 90)),
                                transforms.ToTensor()])

contrast_set = ContrastDataset(img_folder_dataset=test_set,
                               transform=transform,
                               keep_order=True)

# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# network
net = SiameseNet(dim_embedding=config["dim_embedding"],
                 is_rgb=config["is_rgb"])
net.load_state_dict(torch.load("best.siamese.ph"))
net.to(device)
net.eval()

# test
test_set_loader = DataLoader(dataset=contrast_set,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0)

sample_iterator = iter(test_set_loader)
img_query, _, _, img_id_query, _ = sample_iterator.next()
img_id_query = img_id_query[0].split(os.path.sep)[-2]
img_query = img_query.to(device)
embedding_query = net.cal_embedding(img_query)

df_results = pd.DataFrame(columns=["GroudTruth", "TestFace", "Similarity"])

for idx, sample in enumerate(sample_iterator):
    img, _, _, img_id, _ = sample
    img_id = img_id[0].split(os.path.sep)[-2]
    img = img.to(device)
    embedding = net.cal_embedding(img)
    distance = cal_distance(embedding_query, embedding)
    similarity = 1 - distance.item()

    imshow(make_grid(torch.cat((img_query, img), 0).to("cpu")),
           text="Similarity: {:.2f}".format(similarity),
           is_torch=True)

    df_results.loc[idx, :] = [img_id_query, img_id, similarity]

df_results = df_results.sort_values(by="Similarity", ascending=False)
df_results.to_csv("results.csv", index=False, header=True)
print(df_results)