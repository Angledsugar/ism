import torch
from torchvision import models, transforms

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from model.resnet_FT import ResNetGAPFeatures as Net
from utils.data import read_data, create_dataloader, AestheticsDataset

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2
import random

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def extract_pooled_features(inp, net):
    _ = net(inp)
    pooled_features = [features.feature_maps for features in net.all_features] 
    return pooled_features

def downsample_pooled_features(features):
    dim_reduced_features = []
    for pooled_feature in pooled_features:
        if pooled_feature.size()[-1] == 75:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size=(7, 7)))
        elif pooled_feature.size()[-1] == 38:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size = (4, 4), padding=1))
        elif pooled_feature.size()[-1] == 19:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size = (2, 2), padding=1))
        elif pooled_feature.size()[-1] == 10:
            dim_reduced_features.append(pooled_feature)
    dim_reduced_features = torch.cat(dim_reduced_features, dim=1).squeeze()
    return dim_reduced_features

def scale(image, low=-1, high=1):
    im_max = np.max(image)
    im_min = np.min(image)
    return (high - low) * (image - np.min(image))/(im_max - im_min) + low 

def extract_heatmap(features, weights, w, h):
#     cam = np.ones((10, 10), dtype=np.float32) 
    
#     # Sum up the feature maps 
#     temp = weight.view(-1, 1, 1) * features
#     summed_temp = torch.sum(temp, dim=0).data.cpu().numpy()
#     cam = cam + summed_temp
#     cam = cv2.resize(cam, (w, h))
#     cam = np.maximum(cam, 0)
#     cam = np.uint8(255*(cam/np.max(cam)))
    cam = np.zeros((10, 10), dtype=np.float32) 
    temp = weights.view(-1, 1, 1) * downsampled_pooled_features
    summed_temp = torch.sum(temp, dim=0).data.cpu().numpy()
    cam = cam + summed_temp
    cam = cv2.resize(cam, (w, h))
    cam = scale(cam)
    return cam 

train = read_data("./data/train.csv", "./images")
val = read_data("./data/val.csv", "./images")
test = read_data("./data/test.csv", "./images", is_test = True)

train_dataset = AestheticsDataset(train, is_train=False)
val_dataset = AestheticsDataset(val, is_train=False)
test_dataset = AestheticsDataset(test, is_train=False)

use_cuda = torch.cuda.is_available()

save_path = "checkpoint/001" 
checkpoint = "epoch_17.loss_0.3861372387760766.pth"
resnet = models.resnet50(pretrained=True)
net = Net(resnet, n_features=12)

def extract_prediction(inp, net):
    d = dict()
    net.eval()
    output = net(inp)
    for i, key in enumerate(all_keys):
        print(output[:, i])
        d[key] = output[:, i].data[0]
    return d

if use_cuda:
    resnet = resnet.cuda()
    net = net.cuda()
    net.load_state_dict(torch.load(f"{save_path}/{checkpoint}"))
else:
    net.load_state_dict(torch.load(f"{save_path}/{checkpoint}", map_location=lambda storage, loc: storage))
    
attr_keys = ['BalacingElements', 'ColorHarmony', 'Content', 'DoF',
             'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor']
non_neg_attr_keys = ['Repetition', 'Symmetry', 'score']
all_keys = attr_keys + non_neg_attr_keys
used_keys = ["ColorHarmony", "Content", "DoF", "Object", "VividColor", "score"]

weights = {k: net.attribute_weights.weight[i, :] for i, k in enumerate(all_keys)} 

def sample_data(dataset, image_path=None):
    idx = random.sample(range(len(dataset)), 1)[0]
    return dataset[idx]

# Get some test data to see how the heatmaps look
data = sample_data(test_dataset)

image = data['image']
image_path = data['image_path']
image_default = mpimg.imread("image1/image (170).jpg")
img_shape = image_default.shape
h, w = img_shape[0], img_shape[1]

plt.imshow(image_default)

inp = Variable(image).unsqueeze(0)
if use_cuda:
    inp = inp.cuda()
    
predicted_values = extract_prediction(inp, net)
pooled_features = extract_pooled_features(inp, net)
downsampled_pooled_features = downsample_pooled_features(pooled_features)

fig, ax = plt.subplots(figsize=(20, 20))
y, x = np.mgrid[0:h, 0:w]
fig.subplots_adjust(right=1,top=1,hspace=0.5,wspace=0.5)
for i, k in enumerate(used_keys): 
    heatmap = extract_heatmap(downsampled_pooled_features, weights[k], w=w, h=h)
    ax = fig.add_subplot(2, 4, i+1)
    ax.imshow(image_default, cmap='gray')
    cb = ax.contourf(x, y, heatmap, cmap='jet', alpha=0.75)
    ax.set_title(f"Attribute: {k}\nScore: {data[k][0]}\nPredicted Score: {predicted_values[k]}")
    
ax = fig.add_subplot(2, 4, 7)
ax.imshow(image_default) 
plt.colorbar(cb)
plt.tight_layout()

plt.show()