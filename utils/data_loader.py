import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder



class MriSegmentation(data.Dataset):
    """
    Arguments:
        image_path: paths to data folders
        mask_path: paths to mask folders  
        prefix: patient name prefix (optional)
        sample: subset of the data
        targets_path: targets file path
        if ignore_missing (bool): delete subject if the data partially missing
        
    """
    def __init__(self, sample, prefix=False, mask_path=False,
                 image_path='/gpfs/gpfs0/sbi/data/fcd_classification_bank',
                 targets_path='../targets/targets_fcd_bank.csv', ignore_missing=True,
                 coord_min=(20,20,20,), img_shape=(180, 180, 180,),
                 mask ='seg'):
        
        super(MriSegmentation, self).__init__()
        print('Assembling data for: ', sample, ' sample.')

        targets = pd.read_csv(targets_path)
        files = pd.DataFrame(columns = ['patient','scan','fcd','img_file','img_seg'])
        clause = (targets['sample'] == sample)

        if prefix:
            files = pd.DataFrame(columns = ['patient','scan','fcd','img_file','img_seg'])
            clause = (targets['sample'] == sample)&(targets['patient'].str.startswith(prefix))
        if mask_path:
            files['img_mask'] = ''
        
        files['patient']= targets['patient'][clause]
        files['fcd'] = targets['fcd'][clause]
        files['scan'] = targets['scan'][clause]
        
        for i in tqdm(range(len(files))):
            for file_in_folder in glob.glob(os.path.join(image_path,'*norm*')):
                    if (files['patient'].iloc[i] in file_in_folder):
                        files['img_file'].iloc[i] = file_in_folder
            for file_in_folder in glob.glob(os.path.join(image_path,'*aseg*')):
                    if (files['patient'].iloc[i] in file_in_folder):
                        files['img_seg'].iloc[i] = file_in_folder       
            if mask_path:
                for file_in_folder in glob.glob(os.path.join(mask_path,'*.nii*')):
                    files['img_mask'].iloc[i] = file_in_folder
        
        # reindexing an array
        files = files.reset_index(drop=True)
        
        # saving only full pairs of data
        if ignore_missing:
            files.dropna(inplace= True)
        
        le = LabelEncoder()      
        self.img_files = files['img_file']
        self.img_seg = files['img_seg']
        self.scan = le.fit_transform(files['scan'])
        self.scan_keys = le.classes_
        self.target = files['fcd'] 
        
        if mask_path:
            self.img_mask = files['img_mask']
            
        self.coord_min = coord_min
        self.img_shape = img_shape
        self.mask_path = mask_path
        self.mask = mask
        assert mask in ['seg','bb','combined'], "Invalid mask name!"
    
    def reshape_image(img, coord_min, img_shape):
        """
        Function reshapes an image precerving location
        img: np.array 
        coord_min: central coordinates
        img_shape: desired image shape
        """
        img = img[coord_min[0]:coord_min[0] + img_shape[0],
                coord_min[1]:coord_min[1] + img_shape[1],
                coord_min[2]:coord_min[2] + img_shape[2],]
        if img.shape[:3] != img_shape:
            print("Current image shape: {}".format(img.shape[:3]))
            print("Desired image shape: {}".format(img_shape))
            raise AssertionError
        return img.reshape((1,) + img_shape)
    
    def load_nii_to_array(nii_path):
        """ Function returns the data from the *.nii 
            file as np.array()
        """
        return np.asanyarray(nib.load(nii_path).dataobj)    
    
    def __getitem__(self, index):
            img_path = self.img_files[index]
            seg_path = self.img_seg[index]
            
            img_array = load_nii_to_array(img_path)
            seg_array = load_nii_to_array(seg_path)
                                   
            img = reshape_image(img_array, self.coord_min, self.img_shape)
            seg = reshape_image(seg_array, self.coord_min, self.img_shape)
                                   
            if self.mask == 'seg':
                # binarising cortical structures
                seg[seg < 1000] = 0
                seg[seg > 1000] = 1
                return torch.from_numpy(img).float(), torch.from_numpy(seg).float()

            elif self.mask == 'bb':
                # preparing bounding box mask 
                bb_mask_path = self.img_mask[index]
                mask_array = load_nii_to_array(bb_mask_path)
                masked_img = reshape_image(mask_array, self.coord_min, self.img_shape)
                return torch.from_numpy(img).float(), torch.from_numpy(masked_img).float()
            
            elif self.mask == 'combined':
                # binarising cortical structures
                seg[seg < 1000] = 0
                seg[seg > 1000] = 1
                
                # preparing bounding box mask 
                bb_mask_path = self.img_mask[index]
                mask_array = load_nii_to_array(bb_mask_path)
                masked_img = reshape_image(mask_array, self.coord_min, self.img_shape)
                
                # calculating combined mask as intersection of both masks
                comb_mask = np.logical_and(masked_img, seg)
                return torch.from_numpy(img).float(), torch.from_numpy(comb_mask).float()

    def __len__(self):
        return len(self.img_files)