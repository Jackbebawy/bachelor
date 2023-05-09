from dataloader import plydataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from pathlib import Path
import torch.nn.functional as F
import datetime
import logging
from utils import test_semseg
from TSGCNet import TSGCNet
import random
import trimesh
k = 32
test_dataset = plydataset("data/test")
test_loader = DataLoader(test_dataset, batch_size=1)
_, points_face, label_face, label_face_onehot, name, _ = next(iter(test_loader))
coordinate = points_face.transpose(2,1)
coordinate, label_face = Variable(coordinate.float()), Variable(label_face.long())
model = TSGCNet(in_channels=12, output_channels=8, k=k)
# for key in list(checkpoint.keys ()):
#     if 'module.' in key:
#         checkpoint [key.replace('module.','')] = checkpoint[key]
#         del checkpoint [key]
state_dict = torch.load('experiment/maiqi/checkpoints/coordinate_190_0.997875.pth')
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict) 
model.eval()
pred = model(coordinate)
print(pred)
print(pred[0][0])
# Create a Trimesh object from the tensor
pred = pred.detach()
# mesh = trimesh.Trimesh(vertices=pred[0], faces=[])
print(pred.size())
#Save the mesh to a PLY file
# mesh.export('mesh.ply')

# print(next(iter(test_loader)))
# print("points_face ",points_face)
# print("label face ",label_face)