import time

import nibabel as nib

from nilearn import plotting
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_img

import matplotlib.pyplot as plt
import numpy as np
import os

from nipype.interfaces import fsl
from numba import njit


def get_all_patches_and_labels(target_np, gmpm, mask_np, h=16, w=32):
    all_patches = np.ones((1, 2, h, w))
    all_labels = np.array([False])
    for i in range(gmpm.shape[2]):
        slice_gmpm = np.rot90(gmpm[:,:,i])
        slice_target = np.rot90(target_np[:,:,i])
        slice_mask = np.rot90(mask_np[:,:,i])

        for j in range(0, slice_gmpm.shape[0], h):
            subslice_gmpm = slice_gmpm[j:j + h, :]
            subslice_target = slice_target[j:j + h, :]
            subslice_mask = slice_mask[j:j + h, :]

            if subslice_gmpm.sum() == 0.:   #just black stride is useless
                continue

            rodon = subslice_gmpm.sum(0) > 0
            start_idx = rodon.argmax()
            mid_idx = slice_gmpm.shape[1] // 2 - w

            assert start_idx != 0

            #side patches
            patch_1 = np.stack([
                subslice_target[:, start_idx: start_idx + w],
                subslice_target[:, -start_idx-1:-start_idx - w-1:-1]
            ])
            label_1 = subslice_mask[:, start_idx: start_idx + w].sum() > 0

            patch_2 = np.stack([
                subslice_target[:, -start_idx - w : -start_idx],
                subslice_target[:, start_idx + w-1: start_idx-1:-1]
            ])
            label_2 = subslice_mask[:, -start_idx - w : -start_idx].sum() > 0


            if start_idx < mid_idx:
                all_patches = np.concatenate([all_patches, patch_1[None]])
                all_labels = np.concatenate([all_labels, [label_1]])
                all_patches = np.concatenate([all_patches, patch_2[None]])
                all_labels = np.concatenate([all_labels, [label_2]])


            #middle patches
            patch_3 = np.stack([
                subslice_target[:, mid_idx: mid_idx + w],
                subslice_target[:, -mid_idx-1:-mid_idx-1 - w:-1]
            ])
            label_3 = subslice_mask[:, mid_idx: mid_idx + w].sum() > 0

            patch_4 = np.stack([
                subslice_target[:, -mid_idx - w : -mid_idx],
                subslice_target[:, mid_idx - 1 + w : mid_idx - 1 :-1]
            ])
            label_4 = subslice_mask[:,-mid_idx - w : -mid_idx].sum() > 0

            all_patches = np.concatenate([all_patches, patch_3[None]])
            all_labels = np.concatenate([all_labels, [label_3]])
            all_patches = np.concatenate([all_patches, patch_4[None]])
            all_labels = np.concatenate([all_labels, [label_4]])

    #upsampling
    for k in range(1, h):
        for i in range(gmpm.shape[2]):
            slice_gmpm = np.rot90(gmpm[:,:,i])
            slice_target = np.rot90(target_np[:,:,i])
            slice_mask = np.rot90(mask_np[:,:,i])

            for j in range(0, slice_gmpm.shape[0] - h, h):
                subslice_gmpm = slice_gmpm[k + j: k + j + h, :]
                subslice_target = slice_target[k + j: k + j + h, :]
                subslice_mask = slice_mask[k + j: k + j + h, :]

                if subslice_gmpm.sum() == 0.:
                    continue

                rodon = subslice_gmpm.sum(0) > 0
                start_idx = rodon.argmax()
                mid_idx = slice_gmpm.shape[1] // 2 - w

                assert start_idx != 0

                #side patches
                patch_1 = np.stack([
                    subslice_target[:, start_idx: start_idx + w],
                    subslice_target[:, -start_idx-1:-start_idx-1 - w:-1]
                ])
                label_1 = subslice_mask[:, start_idx: start_idx + w].sum() > 0

                patch_2 = np.stack([
                    subslice_target[:, -start_idx - w : -start_idx],
                    subslice_target[:, start_idx -1 + w : start_idx-1 :-1]
                ])
                label_2 = subslice_mask[:, -start_idx - w : -start_idx].sum() > 0

                if start_idx < mid_idx:
                    if label_1:
                        all_patches = np.concatenate([all_patches, patch_1[None]])
                        all_labels = np.concatenate([all_labels, [True]])
                    if label_2:
                        all_patches = np.concatenate([all_patches, patch_2[None]])
                        all_labels = np.concatenate([all_labels, [True]])

                #middle patches
                patch_3 = np.stack([
                    subslice_target[:, mid_idx: mid_idx + w],
                    subslice_target[:, -mid_idx-1:-mid_idx-1 - w:-1]
                ])
                label_3 = subslice_mask[:, mid_idx: mid_idx + w].sum() > 0
                if label_3:
                    all_patches = np.concatenate([all_patches, patch_3[None]])
                    all_labels = np.concatenate([all_labels, [True]])

                patch_4 = np.stack([
                    subslice_target[:, -mid_idx - w : -mid_idx],
                    subslice_target[:, mid_idx -1+ w : mid_idx-1 :-1]
                ])
                label_4 = subslice_mask[:,-mid_idx - w : -mid_idx].sum() > 0
                if label_4:
                    all_patches = np.concatenate([all_patches, patch_4[None]])
                    all_labels = np.concatenate([all_labels, [True]])


    return all_patches[1:], all_labels[1:]

def get_only_patches(target_np, gmpm, h=16, w=32):

    all_patches = np.ones((1, 2, h, w))
    for i in range(gmpm.shape[2]):
        slice_gmpm = np.rot90(gmpm[:,:,i])
        slice_target = np.rot90(target_np[:,:,i])

        for j in range(0, slice_gmpm.shape[0], h):
            subslice_gmpm = slice_gmpm[j:j + h, :]
            subslice_target = slice_target[j:j + h, :]

            if subslice_gmpm.sum() == 0.:   #just black stride is useless
                continue

            rodon = subslice_gmpm.sum(0) > 0
            start_idx = rodon.argmax()
            mid_idx = slice_gmpm.shape[1] // 2 - w

            assert start_idx != 0

            #side patches
            patch_1 = np.stack([
                subslice_target[:, start_idx: start_idx + w],
                subslice_target[:, -start_idx-1:-start_idx - w-1:-1]
            ])

            patch_2 = np.stack([
                subslice_target[:, -start_idx - w : -start_idx],
                subslice_target[:, start_idx + w-1: start_idx-1:-1]
            ])

            if start_idx < mid_idx:
                all_patches = np.concatenate([all_patches, patch_1[None]])
                all_patches = np.concatenate([all_patches, patch_2[None]])

                #middle patches

            patch_3 = np.stack([
                subslice_target[:, mid_idx: mid_idx + w],
                subslice_target[:, -mid_idx-1:-mid_idx-1 - w:-1]
            ])

            patch_4 = np.stack([
                subslice_target[:, -mid_idx - w : -mid_idx],
                subslice_target[:, mid_idx - 1 + w : mid_idx - 1 :-1]
            ])

            all_patches = np.concatenate([all_patches, patch_3[None]])
            all_patches = np.concatenate([all_patches, patch_4[None]])
    return all_patches[1:]

def get_image_patches(input_img_name, input_mask_name=None, h=16, w=32):
    target_img = nib.load(input_img_name)
    target_np = target_img.get_fdata()
    target_np = (target_np - target_np.min()) / (target_np.max() - target_np.min())

    if input_mask_name is not None:
        mask = nib.load(input_mask_name)
        mask_np = mask.get_fdata() > 0
        all_patches, all_labels = get_all_patches_and_labels(target_np, gmpm, mask_np, h=h, w=w)
    else:
        all_patches = get_only_patches(target_np, gmpm, h=h, w=w)
        all_labels = np.zeros(all_patches.shape[0], dtype='bool')
    return all_patches, all_labels
