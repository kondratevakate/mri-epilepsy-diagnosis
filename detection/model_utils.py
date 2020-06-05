import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import os
from IPython import display
%matplotlib inline

!pip install nilearn

from nilearn import plotting
import nibabel as nib
from scipy.signal import convolve


class PatchModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvolutionBlock(2, 16),
            ConvolutionBlock(16, 32),
            ConvolutionBlock(32, 64),
            ConvolutionBlock(64, 128),
            ConvolutionBlock(128, 256),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(3*11*256, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvolutionBlock(nn.Module):
    def __init__(self, in_c, out_c, pad=0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=pad)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def train(n_epochs=20, lr=3e-4, schedule_factor=.1, saving=True):
    model = PatchModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, n_epochs//2, schedule_factor)
    criterion = nn.CrossEntropyLoss()
    n_epochs = n_epochs

    train_loss_history = []
    val_accuracy_history = []
    precision_history = []
    recall_history = []

    best_val_accuracy = 0

    for epoch in range(n_epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            X_batch, y_batch = batch[0].cuda(), batch[1].cuda()

            y_predicted = model(X_batch)
            loss = criterion(y_predicted, y_batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            train_loss_history.append(loss.item())

        scheduler.step()



        model.eval()
        correct = 0
        y_pred = []
        for i, batch in enumerate(val_dataloader):
            X_batch, y_batch = batch[0].cuda(), batch[1].cuda()
            logits = model(X_batch)
            predicted_labels = torch.argmax(logits, dim=1)
            correct += (predicted_labels == y_batch).sum().item()
            y_pred += list(predicted_labels.detach().cpu().numpy())

        y_pred = np.array(y_pred, dtype='bool')

        val_accuracy_history.append(correct/len(val_dataset))
        if val_accuracy_history[-1] > best_val_accuracy and saving:
            torch.save(model.state_dict(), 'best_model.pth')

        precision_history.append(precision_score(y_val, y_pred))
        recall_history.append(recall_score(y_val, y_pred))

        fig, ax = plt.subplots(1, 4, figsize=(16,4))
        ax[0].semilogy(train_loss_history)
        ax[0].set_title('Train loss')
        ax[1].plot(val_accuracy_history, 'r-*')
        ax[1].set_title('Val accuracy', )
        ax[2].plot(precision_history, 'r-*')
        ax[2].set_title('Precision history', )
        ax[3].plot(recall_history, 'r-*')
        ax[3].set_title('Recall history', )
        plt.show()
        display.clear_output(True)
    return model

class FCDMaskGenerator():

    def __init__(self, model_weights='best_model.pth', h=16, w=32):
        self.model = PatchModel().cuda()
        self.model.load_state_dict(torch.load(model_weights))
        self.model.eval()
        self.h = h
        self.w = w
        gray_matter_template = nib.load('MNI152_T1_1mm_brain_gray.nii.gz')
        self.gmpm = gray_matter_template.get_fdata()


    def _infer_patch(self, patch):
        patch_torch = torch.FloatTensor(patch[None]).cuda()
        logits = model(patch_torch)
        label = torch.argmax(logits, dim=1)[0].item()
        return label

    def _get_predictions_per_batches(self, img):
        patch_map_tensor = np.zeros((4, self.gmpm.shape[1]//self.h, self.gmpm.shape[2]), dtype='int64')
        for i in range(self.gmpm.shape[2]):
            slice_gmpm = np.rot90(self.gmpm[:,:,i])
            slice_target = np.rot90(img[:,:,i])
            for j in range(0, slice_gmpm.shape[0], self.h):
                subslice_gmpm = slice_gmpm[j:j + self.h, :]
                subslice_target = slice_target[j:j + self.h, :]
                if subslice_gmpm.sum() == 0.:
                    continue

                rodon = subslice_gmpm.sum(0) > 0
                start_idx = rodon.argmax()
                mid_idx = slice_gmpm.shape[1] // 2 - self.w

                #side patches
                patch_1 = np.stack([
                    subslice_target[:, start_idx: start_idx + self.w],
                    subslice_target[:, -start_idx-1:-start_idx-1 - self.w:-1]
                ])

                patch_2 = np.stack([
                    subslice_target[:, -start_idx - self.w : -start_idx],
                    subslice_target[:, start_idx -1 + self.w : start_idx-1 :-1]
                ])

                if start_idx < mid_idx:
                    patch_map_tensor[0, j//self.h, i] = self._infer_patch(patch_1)
                    patch_map_tensor[3, j//self.h, i] = self._infer_patch(patch_2)

                #middle patches

                patch_3 = np.stack([
                    subslice_target[:, mid_idx: mid_idx + self.w],
                    subslice_target[:, -mid_idx-1:-mid_idx-1 - self.w:-1]
                ])

                patch_4 = np.stack([
                    subslice_target[:, -mid_idx - self.w : -mid_idx],
                    subslice_target[:, mid_idx -1+ self.w : mid_idx-1 :-1]
                ])
               # print('p4', patch_4.shape)

                patch_map_tensor[1, j//self.h, i] = self._infer_patch(patch_3)
                patch_map_tensor[2, j//self.h, i] = self._infer_patch(patch_4)
        return patch_map_tensor

    def _postprocess(self, img, patch_map_tensor):
        count_neighbs = .25*np.array([[
                                [0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]
                            ]])
        res = convolve(patch_map_tensor, count_neighbs, mode='same')
        change_to_pos = (res==1.).astype('int64')
        change_to_neg = (res==0.).astype('int64')
        patch_map_tensor[change_to_pos] = 1
        patch_map_tensor[change_to_neg] = 0
        return patch_map_tensor

    def _masking(self, img, patch_map_tensor):
        final_mask = np.zeros_like(img)
        for i in range(self.gmpm.shape[2]):
            slice_gmpm = np.rot90(self.gmpm[:,:,i])
            slice_target = np.rot90(img[:,:,i])
            for j in range(0, self.gmpm.shape[1], self.h):
                subslice_gmpm = slice_gmpm[j:j + self.h, :]
                subslice_target = slice_target[j:j + self.h, :]
                if subslice_gmpm.sum() == 0.:
                    continue

                rodon = subslice_gmpm.sum(0) > 0
                start_idx = rodon.argmax()
                mid_idx = slice_gmpm.shape[1] // 2 - self.w

                if start_idx < mid_idx:
                    final_mask[start_idx: start_idx + self.w, -j:-j-self.h:-1, i] = patch_map_tensor[0, j//self.h, i]
                    final_mask[-start_idx - self.w : -start_idx, -j:-j-self.h:-1, i] = patch_map_tensor[3, j//self.h, i]
                final_mask[mid_idx: mid_idx + self.w, -j:-j-self.h:-1, i] = patch_map_tensor[1, j//self.h, i]
                final_mask[-mid_idx - self.w : -mid_idx, -j:-j-self.h:-1, i] = patch_map_tensor[2, j//self.h, i]
        return final_mask

    def get_mask(self, img):
        patch_map_tensor = self._get_predictions_per_batches(img)
        patch_map_tensor = self._postprocess(img, patch_map_tensor)
        mask = self._masking(img, patch_map_tensor)
        return mask.astype('int64')

    def get_iou(self, pred_mask, true_mask):
        assert pred_mask.shape == true_mask.shape, 'Wrong shape of masks'
        intersection = np.logical_and(pred_mask, true_mask)
        union = np.logical_or(pred_mask, true_mask)
        return intersection.sum() / union.sum()

   def save_nii_mask(self, mask, img, name='pred_mask.nii.gz'):
       pred_mask_nii = nib.Nifti1Image(mask, img.affine)
       nib.save(pred_mask_nii, name)

    def inference_pipeline(self, input_img_name, input_mask_name=None):
        img = nib.load(input_img_name)
        img_np = img.get_fdata()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        pred_mask_np = self.get_mask(img_np)
        if input_mask_name is not None:
            true_mask = nib.load(input_mask_name)
            true_mask_np = true_mask.get_fdata() > 0
            iou = self.get_iou(pred_mask_np, true_mask_np)
            print('Intersection over union = {:.5f}'.format(iou))

        self.save_nii_mask(pred_mask_np, img,)
