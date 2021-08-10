import os
import random
from albumentations.augmentations import transforms
import pydicom
import numpy as np
import torch
import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd.grad_mode import F
import torch.nn as nn
from torch.nn import functional as torch_functional

from torch.utils import data as torch_data
from efficientnet_pytorch import EfficientNet
import time
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from albumentations.pytorch import ToTensor
from albumentations import Compose, ShiftScaleRotate, Resize

from albumentations import (Cutout, Compose, Normalize, RandomRotate90, HorizontalFlip,
                           VerticalFlip, ShiftScaleRotate, Transpose, OneOf, IAAAdditiveGaussianNoise,
                           GaussNoise, RandomGamma, RandomContrast, RandomBrightness, HueSaturationValue,
                           RandomBrightnessContrast, Lambda, NoOp, CenterCrop, Resize
                           )

train_df = pd.read_csv('./train_labels.csv')

mean_img = [0.22363983, 0.18190407, 0.2523437 ]
std_img = [0.32451536, 0.2956294,  0.31335256]

transform_train = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                         rotate_limit=20, p=0.3, border_mode = cv2.BORDER_REPLICATE),
    Transpose(p=0.5),
    # Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0),
    ToTensor()
])

mri_types = ("FLAIR", "T1w", "T1wCE", "T2w")
# Helper functions to load and process DICOM images
def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)

def get_id(dicomimg):
    return str(dicomimg.SOPInstanceUID)

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(42)

# def get_dicom_metadata(dicomfile):
#     return {
# #         'PatientID': dicomfile.PatientID,
# #         'StudyInstanceID': dicomfile.StudyInstanceID,
# #         'SeriesInstanceID': dicomfile.SeriesInstanceID,
#         'WindowCenter': dicomfile.WindowCenter,
#         'WindowWidth': dicomfile.WindowWidth,
#         'RescaleIntercept': dicomfile.RescaleIntercept,
#         'RescaleSlope': dicomfile.RescaleSlope,
#     }

def prepare_image(imgpath):
    """
        Order: load dicom -> getid -> get metadata -> window image -> normalize img -> load PIL Image.
    """
    pass

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
    
# Windowing algorithm
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

# for i in random.sample(range(train_df.shape[0]), 10):
#     _brats21id = train_df.iloc[i]["BraTS21ID"]
#     _mgmt_value = train_df.iloc[i]["MGMT_value"]
#     visualize_sample(brats21id=_brats21id, mgmt_value=_mgmt_value, slice_i=0.5)
    
class RSNADataset(torch_data.Dataset):
    def __init__(self, paths, targets, input_size, transform=False, label_smoothing=0.1, split='train'):
        self.paths = paths
        self.targets = targets
        self.input_size = input_size
        self.label_smoothing=label_smoothing
        self.rotation = np.random.randint(0,4)
        self.split = split
        self.mri_types = ('FLAIR', 'T1wCE', 'T2w')
        self.transform = transform
        self.transformer = transform_train
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        _bratsid = self.paths[index]
        patient_path = os.path.join('./', self.split, str(_bratsid).zfill(5))
        channels= []
        for i, type in enumerate(self.mri_types, 1):
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, type, "*")),
                key=lambda x: int(x[:-4].split("-")[-1])
            )
            numfiles2 = len(t_paths) // 2
            start = max(0, numfiles2 - 32)
            stop = min(numfiles2*2, numfiles2 + 32)
            for i in range(start, stop):
                dcmimg = load_dicom(t_paths[i], self.input_size, self.rotation)
                img = apply_window_policy(dcmimg)
            
                channels.append(img)
        channels = np.mean(channels, axis=0).transpose(2,0,1)
        # print(channels.shape)
        if self.transform:
            augmented = self.transformer(image=channels)
            channels = augmented['image']
            channels = torch.reshape(channels, (3, 300, 300))
        y = torch.tensor(abs(self.targets[index]-self.label_smoothing), dtype=torch.float)
        # return {"X": torch.tensor(channels).float(), "y": y}
        return {"X": channels.float(), "y": y}
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b0')
        n_features = self.net._fc.in_features
        self.net._dropout = nn.Dropout(p=0.3, inplace=False)
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)
        
    def forward(self, x):
        out = self.net(x)
        return out
    

class Trainer:
    def __init__(self,
                 model,
                 device,
                 criterion,
                 optimizer):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None
        
    def fit(self, epochs, train_loader, val_loader, save_path , patience):
        for n_epoch in range(1, epochs+1):
            self.info_message("EPOCH: {}", n_epoch)
            train_loss, train_time = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time = self.valid_epoch(val_loader)
            
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
        # self.model.eval()

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

def train_model(df_train, df_valid, y_train, y_valid):


    print(df_train.shape, df_valid.shape)
    print(df_train.head())
    
    train_data_retriever = RSNADataset(
        df_train.values, 
        y_train.values,
        input_size=300,
        transform=True,
        split="train/"
    )

    valid_data_retriever = RSNADataset(
        df_valid.values, 
        y_valid.values,
        input_size=300,
        transform=True,
        split="train/"
    )

    train_loader = torch_data.DataLoader(
        train_data_retriever,
        batch_size=12,
        shuffle=True,
        num_workers=0,
        # pin_memory=True
    )

    valid_loader = torch_data.DataLoader(
        valid_data_retriever, 
        batch_size=12,
        shuffle=False,
        num_workers=0,
        # pin_memory=True
    )

    model = Model()
    model.to(device)

    #checkpoint = torch.load("best-model-all-auc0.555.pth")
    #model.load_state_dict(checkpoint["model_state_dict"])

    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00002)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion = torch_functional.binary_cross_entropy_with_logits

    trainer = Trainer(
        model, 
        device, 
        criterion, 
        optimizer
    )

    history = trainer.fit(
        300, 
        train_loader, 
        valid_loader, 
        "./1108", 
        100,
    )
    
    return trainer.lastmodel

modelfiles = None

X_train, X_val, y_train, y_val = train_test_split(train_df['BraTS21ID'], train_df['MGMT_value'], test_size=0.3, random_state=42, stratify=train_df['MGMT_value'])

# dataset = RSNADataset(paths=X_train.values,
#                   targets=y_train.values,
#                   input_size=224,
#                   split="train/")
# # print(dataset[1]['X'].shape)

if not modelfiles:
    modelfiles = train_model(X_train, X_val, y_train, y_val)
    print(modelfiles)
    
    