import os
import random
from re import M
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

from windowing import *



device = "cuda" if torch.cuda.is_available() else "cpu"

def predict(modelfile, df, split):
    print("Predict:", modelfile, df.shape)
    data_retriever = RSNADataset(
        df['BraTS21ID'].values, 
        df["MGMT_value"].values,
        input_size=224,
        split=split
    )

    data_loader = torch_data.DataLoader(
        data_retriever,
        batch_size=12,
        shuffle=False,
        num_workers=8,
    )
   
    model = Model()
    model.to(device)
    
    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    y_pred = []
    ids = []

    for e, batch in enumerate(data_loader,1):
        print(f"{e}/{len(data_loader)}", end="\r")
        with torch.no_grad():
            # print(batch)
            tmp_pred = torch.sigmoid(model(batch["X"].to(device))).cpu().numpy().squeeze()
            # print(tmp_pred)
            if tmp_pred.size == 1:
                y_pred.append(tmp_pred)
            else:
                y_pred.extend(tmp_pred.tolist())
            # ids.extend(batch["id"].numpy().tolist())
            
    preddf = pd.DataFrame({"BraTS21ID": df['BraTS21ID'], "MGMT_value": y_pred}) 
    preddf = preddf.set_index("BraTS21ID")
    return preddf

def inference(mri_types, input_size, modelfile):
    model = Model()
    model.to(device)
    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sub_df = pd.read_csv('./sample_submission.csv')
    preds = []
    for bratsid in sub_df['BraTS21ID'].values:
        patient_path = os.path.join('./test/', str(bratsid).zfill(5))
        channels = []
        for i, type in enumerate(mri_types, 1):
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, type, "*")),
                key=lambda x: int(x[:-4].split("-")[-1])
            )
            ch = len(t_paths)
            for i in range(len(t_paths)):
                dcmimg = load_dicom(t_paths[i], input_size)
                img = apply_window_policy(dcmimg)
                
                channels.append(img)
        channels = np.mean(channels, axis=0).transpose(2,0,1)
        channels = torch.tensor(channels).float().to(device).unsqueeze(0)
        print(channels.shape)
        
        pred = torch.sigmoid(model(channels)).cpu().detach().numpy().squeeze()
        preds.append(pred)
    sub_df['MGMT_value'] = preds
    sub_df = sub_df.set_index('BraTS21ID')
    sub_df.to_csv('final_submission.csv')
    

submission = pd.read_csv("./sample_submission.csv")

submission["MGMT_value"] = 0
mdf = './1108-e3-loss0.693-auc0.645.pth'

# pred = predict(mdf, submission, split="test/")
# submission["MGMT_value"] += pred["MGMT_value"]

# submission["MGMT_value"].to_csv("submission1.csv")
inference(("FLAIR", "T1wCE", "T2w"), 224, mdf)