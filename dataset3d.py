import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import glob
import os
from tqdm import tqdm
import torch
from torch.utils import data as torch_data
import torch.nn as nn
import cv2
from PIL import Image
import re
import matplotlib.pyplot as plt
import pandas as pd
import time
import collections
import torch.nn.functional as F
from torch.nn import functional as torch_functional
from sklearn.metrics import roc_auc_score
import random
from sklearn.model_selection import train_test_split

from efficientnet_pytorch_3d import EfficientNet3D  

IS_NOTEBOOK = False
SIZE=224
NUM_IMAGES=64
MRI_TYPES = ('FLAIR', 'T1w', 'T1wCE', 'T2w')

train_df = pd.read_csv('./train_labels.csv')
X_train, X_val, y_train, y_val = train_test_split(train_df['BraTS21ID'], train_df['MGMT_value'], test_size=0.3, random_state=42, stratify=train_df['MGMT_value'])



data_directory = '../input/kaggle-rsna-brain-tumor-radiogenomic-classification/' if IS_NOTEBOOK else './'

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(42)

def load_dicom(path, img_size, rotate=0):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    
    if rotate > 0:
        rot_choices = [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
        data = cv2.rotate(data, rot_choices[rotate])
    data = cv2.resize(data, (img_size, img_size))
    return data


def load_dicom_image(path, img_size=SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    if rotate > 0:
        rot_choices = [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
        data = cv2.rotate(data, rot_choices[rotate])
        
    data = cv2.resize(data, (img_size, img_size))
    return data


def load_dicom_images_3d(scan_id, num_imgs=NUM_IMAGES, img_size=SIZE, mri_type="FLAIR", split="train/", rotate=0):
    patient_path = os.path.join('./', split, str(scan_id).zfill(5))
    files = sorted(
        glob.glob(os.path.join(patient_path, mri_type, "*")),
        key=lambda x: int(x[:-4].split("-")[-1])
    )
    middle = len(files)//2
    num_imgs2 = num_imgs//2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)
    img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2]]).T 
    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
        img3d = np.concatenate((img3d,  n_zero), axis = -1)
        
    if np.min(img3d) < np.max(img3d):
        img3d = img3d - np.min(img3d)
        img3d = img3d / np.max(img3d)
    # return np.expand_dims(img3d, 0)
    return img3d

def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image

# Windowing policy is to sample brain -> subdural -> bone window then merge to create a 3 channel image
def apply_window_policy(image):
    image1 = apply_window(image, 40, 80) # -> brain window
    image2 = apply_window(image, 80, 200) # -> subdural window
    image3 = apply_window(image, 40, 380) # -> bone window
    image1 = image1 / 80
    image2 = (image2 + 20) / 200
    image3 = (image3 + 150) / 380
    final_image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)
    return final_image

# a = load_dicom_images_3d("00000")
# print(a.shape)
# print(np.min(a), np.max(a), np.mean(a), np.median(a))


class RSNADataset3D(torch_data.Dataset):
    def __init__(self,
                 paths,
                 targets,
                 transform=False,
                 split='train/',
                 input_size=224):
        self.paths = paths
        self.targets = targets
        self.transform=transform
        self.split = split
        self.input_size = input_size
        self.transformer = None
        self.rotation = np.random.randint(0,4)
        self.mri_types = ("FLAIR", "T1w", "T1wCE", "T2w")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        _id = self.paths[index]
        files = os.path.join('./', self.split, str(_id).zfill(5))
        channels = []
        # img3d = load_dicom_images_3d(_id, mri_type="FLAIR")
        for i, type in enumerate(self.mri_types, 1):
            img3d = load_dicom_images_3d(_id, mri_type=type) 
            # img3d = apply_window_policy(img3d)
            channels.append(img3d)
        channels = np.mean(channels, axis=0)
        channels = np.stack([channels, channels, channels], axis=0)
        if len(self.targets):
            y = torch.tensor(self.targets[index], dtype=torch.float)
            return {"X": torch.tensor(channels).float(), "y": y}
        else:
            return {"X": torch.tensor(channels).float(), "y": _id}
        
# dataset = RSNADataset3D(paths=X_train.values,
#                   targets=y_train.values,
#                   input_size=224,
#                   transform=True,
#                   split="train/")
# print(dataset[1]['X'].shape)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet3D.from_name("efficientnet-b0")
        n_features = self.net._fc.in_features
        # self.net._dropout = nn.Dropout(p=0.2, inplace=False)
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)
    
    def forward(self, x):
        out = self.net(x)
        return out
    
class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None
        
    def fit(self, epochs, train_loader, valid_loader, save_path, patience):        
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)
            
            train_loss, train_time = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time = self.valid_epoch(valid_loader)
            
            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s            ",
                n_epoch, train_loss, train_time
            )
            
            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
                n_epoch, valid_loss, valid_auc, valid_time
            )

            # if True:
            # if self.best_valid_score < valid_auc: 
            if self.best_valid_score > valid_loss: 
                self.save_model(n_epoch, save_path, valid_loss, valid_auc)
                self.info_message(
                     "auc improved from {:.4f} to {:.4f}. Saved model to '{}'", 
                    self.best_valid_score, valid_loss, self.lastmodel
                )
                self.best_valid_score = valid_loss
                self.n_patience = 0
            else:
                self.n_patience += 1
            
            if self.n_patience >= patience:
                self.info_message("\nValid auc didn't improve last {} epochs.", patience)
                break
            
    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        sum_loss = 0

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)
            
            loss = self.criterion(outputs, targets)
            loss.backward()

            sum_loss += loss.detach().item()

            self.optimizer.step()
            
            message = 'Train Step {}/{}, train_loss: {:.4f}'
            self.info_message(message, step, len(train_loader), sum_loss/step, end="\r")
        
        return sum_loss/len(train_loader), int(time.time() - t)
    
    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()
                y_all.extend(batch["y"].tolist())
                outputs_all.extend(outputs.tolist())

            message = 'Valid Step {}/{}, valid_loss: {:.4f}'
            self.info_message(message, step, len(valid_loader), sum_loss/step, end="\r")
            
        y_all = [1 if x > 0.5 else 0 for x in y_all]
        auc = roc_auc_score(y_all, outputs_all)
        
        return sum_loss/len(valid_loader), auc, int(time.time() - t)
    
    def save_model(self, n_epoch, save_path, loss, auc):
        self.lastmodel = f"{save_path}-e{n_epoch}-loss{loss:.3f}-auc{auc:.3f}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.lastmodel,
        )
    
    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mri_type(X_train, X_val, y_train, y_val):
    # if mri_type=="all":
    #     train_list = []
    #     valid_list = []
    #     for mri_type in MRI_TYPES:
    #         df_train.loc[:,"MRI_Type"] = mri_type
    #         train_list.append(df_train.copy())
    #         df_valid.loc[:,"MRI_Type"] = mri_type
    #         valid_list.append(df_valid.copy())

    #     df_train = pd.concat(train_list)
    #     df_valid = pd.concat(valid_list)
    # else:
    #     df_train.loc[:,"MRI_Type"] = mri_type
    #     df_valid.loc[:,"MRI_Type"] = mri_type

    print(X_train.shape, X_val.shape)
    print(X_train.head())
    train_data_retriever = RSNADataset3D(
        X_train.values, 
        y_train.values, 
    )

    valid_data_retriever = RSNADataset3D(
        X_val.values, 
        y_val.values,
    )

    train_loader = torch_data.DataLoader(
        train_data_retriever,
        batch_size=4,
        shuffle=True,
        num_workers=8,
    )

    valid_loader = torch_data.DataLoader(
        valid_data_retriever, 
        batch_size=4,
        shuffle=False,
        num_workers=8,
    )

    model = Model()
    model.to(device)

    #checkpoint = torch.load("best-model-all-auc0.555.pth")
    #model.load_state_dict(checkpoint["model_state_dict"])

    #print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.00002)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion = torch_functional.binary_cross_entropy_with_logits

    trainer = Trainer(
        model, 
        device, 
        optimizer, 
        criterion
    )

    history = trainer.fit(
        5, 
        train_loader, 
        valid_loader, 
        "./weights/12083D-", 
        5,
    )
    
    return trainer.lastmodel

modelfiles = None

if not modelfiles:
    modelfiles = train_mri_type(X_train, X_val, y_train, y_val)
    print(modelfiles)