import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm


def load_nii_to_array(nii_path):
    return nib.load(nii_path).get_data()


class LA5_Siblings_MRI(data.Dataset):
    """
    Arguments:
        paths: paths to data folders
        target_path: path to file with targets and additional information
        load_online (bool): if True, load mri images online. Else, preload everything during initialization
    """

    def __init__(self, paths, target_path, load_online=False, mri_type="sMRI",
                 mri_file_suffix="", brain_mask_suffix=None, coord_min=(20, 20, 20,),
                 img_shape=(152, 188, 152,), fixed_start_pos=None, seq_len=None, problems=None, transform=None,
                 temp_storage_path=None):

        self.load_online = load_online
        self.mri_type = mri_type
        self.coord_min = coord_min
        self.img_shape = img_shape
        self.fixed_start_pos = fixed_start_pos
        self.seq_len = seq_len
        self.problems = problems
        self.transform = transform
        self.temp_storage_path = temp_storage_path
        
        if self.problems != None:
            assert len(self.problems) == 1 # more is not supported now

        self.mri_paths = {
            "participant_id": [],
            "path": [],
        }
        self.brain_mask_paths = {
            "participant_id": [],
            "mask_path": [],
        }

        for path_to_folder in paths:
            for patient_folder_name in os.listdir(path_to_folder):
                if 'sub-' in patient_folder_name:
                    path_to_patient_folder = path_to_folder + patient_folder_name
                    mri_type_folder_name = "anat" if self.mri_type == "sMRI" else "func"
                    if os.path.isdir(path_to_patient_folder) and mri_type_folder_name in os.listdir(
                            path_to_patient_folder):
                        temp_path = path_to_patient_folder + "/" + mri_type_folder_name + "/"
                        for filename in os.listdir(temp_path):
                            full_path = temp_path + filename
                            if mri_file_suffix in filename:
                                self.mri_paths["participant_id"].append(patient_folder_name)
                                self.mri_paths["path"].append(full_path)
                            if brain_mask_suffix is not None and brain_mask_suffix in filename:
                                self.brain_mask_paths["participant_id"].append(patient_folder_name)
                                self.brain_mask_paths["mask_path"].append(full_path)

        self.mri_paths = pd.DataFrame(self.mri_paths)
        if brain_mask_suffix is not None:
            self.brain_mask_paths = pd.DataFrame(self.brain_mask_paths)
            self.mri_paths = self.mri_paths.merge(self.brain_mask_paths, on="participant_id")

        if target_path is None:
            if brain_mask_suffix is not None:
                self.brain_mask_paths = self.mri_paths["mask_path"].tolist()
            self.mri_paths = self.mri_paths["path"].tolist()
        else:
            target_df = pd.read_csv(target_path)
            target_df = target_df.merge(self.mri_paths, on="participant_id")
            assert self.problems != None
            #             target_df.dropna(subset=problems, how='any', inplace=True)
            target_df.dropna(subset=problems, how='all', inplace=True)
            target_df.fillna(value=-100,
                             inplace=True)  # -100 default value for ignore_index in cross-entropy loss in PyTorch
            target_df.reset_index(drop=True, inplace=True)
            self.labels = target_df[problems].astype('long').values
            if self.labels.shape[1] == 1:
                self.labels = self.labels.squeeze()
            self.mri_paths = target_df["path"].tolist()
            self.pids = target_df["participant_id"].values
            assert len(set(self.pids)) == len(self.pids)
            if brain_mask_suffix is not None:
                self.brain_mask_paths = target_df["mask_path"].tolist()
            del target_df

        if not self.load_online:
            self.mri_files = [self.get_image(i) for i in tqdm(range(len(self.mri_paths)))]

    def reshape_image(self, img, coord_min, img_shape):
        img = img[
              coord_min[0]:coord_min[0] + img_shape[0],
              coord_min[1]:coord_min[1] + img_shape[1],
              coord_min[2]:coord_min[2] + img_shape[2],
              ]
        if img.shape[:3] != img_shape:
            print("Current image shape: {}".format(img.shape[:3]))
            print("Desired image shape: {}".format(img_shape))
            raise AssertionError
        if self.mri_type == "sMRI":
            img = img.reshape((1,) + img_shape)
        elif self.mri_type == "fMRI":
            seq_len = img.shape[-1]
            img = img.reshape((1,) + img_shape + (seq_len,))
        return img

    def get_image(self, index):
        def load_mri(mri_path):
            if "nii" in mri_path:
                if self.temp_storage_path is not None:
                    if not os.path.exists(self.temp_storage_path):
                        os.makedirs(self.temp_storage_path)
                    temp_file_path = self.temp_storage_path + mri_path.split('/')[-1].split('.')[0] + '.npy'
                    try:
                        img = np.load(temp_file_path)  # 1s
                    except FileNotFoundError:
                        img = load_nii_to_array(mri_path)  # 2.5s
                        np.save(temp_file_path, img)
                else:
                    img = load_nii_to_array(mri_path)  # 2.5s
            else:
                img = np.load(mri_path)  # 1s
            return img

        img = load_mri(self.mri_paths[index])

        try:
            brain_mask = load_mri(self.brain_mask_paths[index])
            if self.mri_type == "fMRI":
                brain_mask = brain_mask[..., np.newaxis]
            img *= brain_mask
            del brain_mask
        except KeyError:
            pass

        img = self.reshape_image(img, self.coord_min, self.img_shape)

        if self.mri_type == "fMRI":
            assert self.seq_len != None and self.seq_len > 0
            start_pos = np.random.choice(
                img.shape[-1] - self.seq_len) if self.fixed_start_pos is None else self.fixed_start_pos
            img = img[:, :, :, :, start_pos:start_pos + self.seq_len]
            assert img.shape[-1] == self.seq_len

        return img

    def __getitem__(self, index):
        if not self.load_online:
            item = self.mri_files[index]
        else:
            item = self.get_image(index)

        if self.mri_type == "fMRI":
            item = np.moveaxis(item, -1, 0)
        if self.transform is not None:
            item = self.transform(item)
        if self.problems is not None:
            return (item, self.labels[index])
        return (item, None)

    def __len__(self):
        return len(self.mri_paths)
