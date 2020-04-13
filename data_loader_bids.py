import os
import copy
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from utils import *

class BidsMRI(data.Dataset):
    """
    Arguments:
        path: path to data folder
        labels_path: path to file with targets and additional information
        target: column of targets df with target to predict. If None, loads images only
        encode_target: if True, encode target with LabelEncoder
        load_online (bool): if True, load mri images online. Else, preload everything during initialization
    """
    
    def __init__(self, paths, labels_path, target=None, encode_target=False, load_online=False, 
                 mri_type="sMRI", mri_file_suffix="", brain_mask_suffix=None, transform=None,
                 coord_min=(20, 20, 20,), img_shape=(152, 188, 152,), start_pos=None, seq_len=None):
        self.mri_paths = {
            "participant_id" : [],
            "path" : [],
        }
        
        self.paths = paths if type(paths) is list else [paths]
        self.labels = pd.read_csv(labels_path)
        self.target = self.set_target(target, encode_target)
        self.load_online = load_online
        
        self.mri_type = mri_type
        if self.mri_type == "sMRI":
            self.type = "anat" 
        elif self.mri_type == "fMRI":
            self.type = "func"
        else:
            self.type = None
#             raise ValueError("Select sMRI or fMRI mri type.")
        self.mri_file_suffix = mri_file_suffix
        
        self.brain_mask_suffix = brain_mask_suffix
        if brain_mask_suffix is not None:
            self.brain_mask_paths = {
                "participant_id" : [],
                "mask_path" : [],
            }
    
        self.coord_min = coord_min
        self.img_shape = img_shape
        self.start_pos = start_pos
        self.seq_len = seq_len
        self.transform = transform
        
        for path_to_folder in self.paths:
            for patient_folder_name in os.listdir(path_to_folder):
                if 'sub-' in patient_folder_name and os.path.isdir(path_to_folder + patient_folder_name):

                    if self.type is not None and self.type in os.listdir(path_to_folder + patient_folder_name):
                        temp_path = path_to_folder + patient_folder_name + "/" + self.type + "/"
                    elif self.type is None:
                        temp_path = path_to_folder + patient_folder_name + "/"
                    else:
                        continue

                    for filename in os.listdir(temp_path):
                        if self.mri_file_suffix in filename:
                            self.mri_paths["participant_id"].append(patient_folder_name)
                            full_path = temp_path + filename
                            self.mri_paths["path"].append(full_path)
                        if self.brain_mask_suffix is not None and self.brain_mask_suffix in filename:
                            self.brain_mask_paths["participant_id"].append(patient_folder_name)
                            full_path = temp_path + filename
                            self.brain_mask_paths["mask_path"].append(full_path)
                            
        self.mri_paths = pd.DataFrame(self.mri_paths)
        self.labels = self.labels.merge(self.mri_paths, on="participant_id")
        self.mri_files = self.labels["path"].tolist()
        
        self.brain_mask_files = None
        if self.brain_mask_suffix is not None:
            self.brain_mask_paths = pd.DataFrame(self.brain_mask_paths)
            self.labels = self.labels.merge(self.brain_mask_paths, on="participant_id")
            self.brain_mask_files = self.labels["mask_path"].tolist()
        
        if not self.load_online:
            self.mri_files = [self.get_image(index, self.start_pos, self.seq_len) for index in tqdm(range(len(self.mri_files)))]

        # update self.img_shape (and other params ?)
        self.output_img_shape = self[0].shape[1:4]
        
    
    def set_target(self, target=None, encode_target=False):
        if target is not None:
            self.target = self.labels[target]
            if encode_target:
                enc = LabelEncoder()
                idx = self.target.notnull()
                self.target[idx] = enc.fit_transform(self.target[idx])
        else:
            self.target = None
        return self.target

            
    def reshape_image(self, mri_img, coord_min, img_shape):
        if self.mri_type == "sMRI":
            return mri_img[coord_min[0]:coord_min[0] + img_shape[0],
                           coord_min[1]:coord_min[1] + img_shape[1],
                           coord_min[2]:coord_min[2] + img_shape[2]].reshape((1,) + img_shape)
        if self.mri_type == "fMRI":
            seq_len = mri_img.shape[-1]
            return mri_img[coord_min[0]:coord_min[0] + img_shape[0],
                           coord_min[1]:coord_min[1] + img_shape[1],
                           coord_min[2]:coord_min[2] + img_shape[2], :].reshape((1,) + img_shape + (seq_len,))
        
    def get_image(self, index, start_pos=None, seq_len=None):
        
        def load_mri(mri_file):
            if "nii" in mri_file:
                img = load_nii_to_array(mri_file)
            else:
                img = np.load(mri_file)
            return img
        
        mri_file = self.mri_files[index]
        img = load_mri(mri_file)
        
        if self.brain_mask_files is not None:
            brain_mask_file = self.brain_mask_files[index]
            img *= load_mri(brain_mask_file)[..., np.newaxis]
            
        img = self.reshape_image(img, self.coord_min, self.img_shape)
        
        if self.mri_type == "sMRI":
            return img
        
        if self.mri_type == "fMRI":
            if seq_len is None:
                seq_len = img.shape[-1]
            # what if seq_len == 0 ?
            if start_pos is None:
                start_pos = np.random.choice(img.shape[-1] - seq_len)
            if seq_len == 1:
                img = img[:, :, :, :, start_pos]
            else:
                img = img[:, :, :, :, start_pos:start_pos + seq_len]
                img = img.transpose((4, 0, 1, 2, 3))
            return img
    
    def __getitem__(self, index):
        img = self.get_image(index, self.start_pos, self.seq_len) if self.load_online else self.mri_files[index]
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.target[index]) if self.target is not None else img
    
    def __len__(self):
        return len(self.mri_files)


# таргеты возвращаются в том же порядке, с _теми же_ индексами
# но, видимо, в процессе обучения мы будем брать только те индексы, которые соответствуют нотналл позициям здесь
# те для списка с данными (упорядоченного в том же порядке, что и общий список индексов)
# ничего не меняется в зависимости от задачи, 
# варьируется только то, какое _подмножество индексов_ мы используем для получения данных для обучения