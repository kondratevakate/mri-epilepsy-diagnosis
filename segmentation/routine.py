import sys
import os
import warnings
import time
import random
import numpy as np
from optparse import OptionParser


import enum
from tqdm import tqdm_notebook, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.vgg import vgg11_bn
from torch.autograd import Function, Variable
from torch.utils.data import DataLoader, Subset
import torch.backends.cudnn as cudnn
from torch import optim
import multiprocessing

from IPython.display import clear_output
from sklearn.model_selection import StratifiedKFold, ShuffleSplit

import torchio
from unet import UNet
from torch.utils.data import DataLoader, Subset
from torchio import AFFINE, DATA, PATH, TYPE, STEM
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)


"""
    Code adapted from: https://github.com/fepegar/torchio#credits

        Credit: Pérez-García et al., 2020, TorchIO: 
        a Python library for efficient loading, preprocessing, 
        augmentation and patch-based sampling of medical images in deep learning.

"""

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

MRI = 'MRI'
LABEL = 'LABEL'

class Action(enum.Enum):
    TRAIN = 'Training'
    VALIDATE = 'Validation'

def get_torchio_dataset(inputs, targets, transform):
    """
    The function creates dataset from the list of files from cunstumised dataloader.
    """
    subjects = []
    for (image_path, label_path) in zip(inputs, targets ):
        subject_dict = {
            MRI : torchio.Image(image_path, torchio.INTENSITY),
            LABEL: torchio.Image(label_path, torchio.LABEL),
        }
        subject = torchio.Subject(subject_dict)
        subjects.append(subject)
    
    if transform:
        dataset = torchio.ImagesDataset(subjects, transform = transform)
    elif not transform:
        dataset = torchio.ImagesDataset(subjects)
    
    return  dataset

def get_loaders(data, cv_split,
        training_transform = False,
        validation_transform = False,
        patch_size = 64,
        patches = False,
        samples_per_volume = 6,
        max_queue_length = 180,
        training_batch_size = 1,
        validation_batch_size = 1):
    
    """
    The function creates dataloaders 
    
        weights_stem (str): ['full_size', 'patches'] #sizes of training objects
        transform (bool): False # data augmentation
        batch_size (int): 1 # batch sizes for training
        
    """
    
    training_idx, validation_idx = cv_split
    
    print('Training set:', len(training_idx), 'subjects')
    print('Validation set:', len(validation_idx), 'subjects')
    
    training_set = get_torchio_dataset(
        list(data.img_files[training_idx].values), 
        list(data.img_seg[training_idx].values),
        training_transform 
    )
    validation_set = get_torchio_dataset(
        list(data.img_files[validation_idx].values), 
        list(data.img_seg[validation_idx].values),
        validation_transform
    )
    
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=training_batch_size)

    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=validation_batch_size)
    
    if patches:

        patches_training_set = torchio.Queue(
            subjects_dataset=training_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            patch_size=patch_size,
            sampler_class=torchio.sampler.ImageSampler,
            num_workers=multiprocessing.cpu_count(),
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        patches_validation_set = torchio.Queue(
            subjects_dataset=validation_set,
            max_length=max_queue_length,
            samples_per_volume=samples_per_volume,
            patch_size=patch_size,
            sampler_class=torchio.sampler.ImageSampler,
            num_workers=multiprocessing.cpu_count(),
            shuffle_subjects=False,
            shuffle_patches=False,
        )

        training_loader = torch.utils.data.DataLoader(
            patches_training_set, batch_size=training_batch_size)

        validation_loader = torch.utils.data.DataLoader(
            patches_validation_set, batch_size=validation_batch_size)
        
        print('Training loader length:', len(training_loader))
        print('Validation loader length:', len(validation_loader))
    
    return training_loader, validation_loader

def prepare_batch(batch, device):
    """
    The function loaging *nii.gz files, sending to the devise.
    For the LABEL in binarises the data.
    """
    inputs = batch[MRI][DATA].to(device)
    targets = batch[LABEL][DATA]
    targets[targets < 1000] = 0
    targets[targets > 1000] = 1
    targets = targets.to(device)    
    return inputs, targets

def get_iou_score(prediction, ground_truth):
    n_images = len(prediction)
    intersection, union = 0, 0
    for i in range(n_images):
        intersection += np.logical_and(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum() 
        union += np.logical_or(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum()
    iou_score = float(intersection) / union
    return iou_score

def get_dice_score(output, target, SPATIAL_DIMENSIONS = (2, 3, 4), epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return dice_score

def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)

def forward(model, inputs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        logits = model(inputs)
    return logits

def run_epoch(epoch_idx, action, loader, model, optimizer, scheduler, experiment= False):
    is_training = action == Action.TRAIN
    epoch_losses = []
    model.train(is_training)
    
    for batch_idx, batch in enumerate(tqdm(loader)):
        inputs, targets = prepare_batch(batch, device)
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_training):
            logits = forward(model, inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            batch_losses = get_dice_loss(probabilities, targets)
            batch_loss = batch_losses.mean()
            
            if is_training:
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                
            # appending the loss
            epoch_losses.append(batch_loss.item())
           
            if experiment:
                if action == Action.TRAIN:
                    experiment.log_metric("train_dice_loss", batch_loss.item())
                elif action == Action.VALIDATE:
                    experiment.log_metric("validate_dice_loss", batch_loss.item())
                    
            del inputs, targets, logits, probabilities, batch_losses
    
    epoch_losses = np.array(epoch_losses)
#     print(f'{action.value} mean loss: {epoch_losses.mean():0.3f}')
    
    return epoch_losses 

def train(num_epochs, training_loader, validation_loader, model, optimizer, scheduler,
          weights_stem, save_epoch= 2, experiment= False, verbose = True):
    
    start_time = time.time()
    epoch_train_loss, epoch_val_loss = [], []
    
    run_epoch(0, Action.VALIDATE, validation_loader, model, optimizer, scheduler, experiment)
    
    for epoch_idx in range(1, num_epochs + 1):
        
        epoch_train_losses = run_epoch(epoch_idx, Action.TRAIN, training_loader, 
                                       model, optimizer,scheduler, experiment)
        epoch_val_losses = run_epoch(epoch_idx, Action.VALIDATE, validation_loader, 
                                     model, optimizer, scheduler, experiment)
        
        # 4. Print metrics
        if verbose:
            clear_output(True)
            print("Epoch {} of {} took {:.3f}s".format(epoch_idx, num_epochs, time.time() - start_time))
            print("  training loss (in-iteration): \t{:.6f}".format(epoch_train_losses[-1]))
            print("  validation loss: \t\t\t{:.6f}".format(epoch_val_losses[-1]))    
        
        epoch_train_loss.append(np.mean(epoch_train_losses))
        epoch_val_loss.append(np.mean(epoch_val_losses))
        
        # 5. Plot metrics
        if verbose:
            plt.figure(figsize=(10, 5))
            plt.plot(epoch_train_loss, label='train')
            plt.plot(epoch_val_loss, label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
    
        if experiment:
            experiment.log_epoch_end(epoch_idx)
        if (epoch_idx% save_epoch == 0):
            torch.save(model.state_dict(), f'weights/{weights_stem}_epoch_{epoch_idx}.pth')
            
def get_model_and_optimizer(device, num_encoding_blocks=3, out_channels_first_layer=8, step_size=3):
    
    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=num_encoding_blocks,
        out_channels_first_layer=out_channels_first_layer,
        normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    
    return model, optimizer, scheduler