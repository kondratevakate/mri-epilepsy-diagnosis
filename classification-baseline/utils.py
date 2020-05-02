import os
import numpy as np
import torch
import pydicom
import nibabel as nib
import nibabel.processing
import matplotlib.pyplot as plt

# def load_nii_to_array(nii_path,voxel_size = [2, 2, 2]):
#     file_ = nib.load(nii_path)
#     resampled_img = nibabel.processing.resample_to_output(file_, voxel_size)
#     return resampled_img.get_data()

def load_nii_to_array(nii_path):
    return nib.load(nii_path).get_data()

def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
        
def save_res(res, path):
    ensure_dir(path)
    with open(path, "w") as f:
        f.write(str(res))
        
def load_res(path):
    with open(path) as f:
        res = f.read()
    return eval(res)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    state - dict containing:
    "model" : model.state_dict(),
    "optimizer" : optimizer.state_dict(),
    (optionally) loss, epoch, etc.
    """
    ensure_dir(filename)
    torch.save(state, filename)
    
def load_checkpoint(filename):
    """
    state - dict containing:
    "model" : model.state_dict(),
    "optimizer" : optimizer.state_dict()
    """
    state = torch.load(filename)
    return state
    
# def load_checkpoint(filename):
#     """
#     """
# #     model = TheModelClass(*args, **kwargs)
# #     optimizer = TheOptimizerClass(*args, **kwargs)

#     checkpoint = torch.load(filename)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     opt.load_state_dict(checkpoint['optimizer_state_dict'])
# #     epoch = checkpoint['epoch']
# #     loss = checkpoint['loss']
    
    
def load_results(name, problem, metric="auc"):
    train_loss_l = load_res("models/{}/{}/train_loss".format(
        name, problem.replace("/", "_")))
    val_loss_l = load_res("models/{}/{}/val_loss".format(
        name, problem.replace("/", "_")))
    train_metric_l = load_res("models/{}/{}/train_{}".format(
        name, problem.replace("/", "_"), metric))
    val_metric_l = load_res("models/{}/{}/val_{}".format(
        name, problem.replace("/", "_"), metric))
#     val_last_preds_l = load_res("models/" + problem_name + "/val_last_probs_" + problem.replace("/", "_"))
    return train_loss_l, val_loss_l, train_metric_l, val_metric_l #, val_last_preds_l
    
def save_results(name, problem, 
                 train_loss_l=[], 
                 val_loss_l=[], 
                 train_metric_l=[], 
                 val_metric_l=[], 
                 val_last_preds_l=None,
                 metric="auc"):
    save_res(train_loss_l, "models/{}/{}/train_loss".format(
        name, problem.replace("/", "_")))
    save_res(val_loss_l, "models/{}/{}/val_loss".format(
        name, problem.replace("/", "_")))
    save_res(train_metric_l, "models/{}/{}/train_{}".format(
        name, problem.replace("/", "_"), metric))
    save_res(val_metric_l, "models/{}/{}/val_{}".format(
        name, problem.replace("/", "_"), metric))
    if val_last_preds_l is not None:
        raise NotImplementedError
    print("saved.")

    
def plot_losses(problem_name, problem, mean=False, metric="auc"):
    train_loss_l, val_loss_l, train_metric_l, val_metric_l = load_results(problem_name, problem, metric)
    if mean:
        plt.figure(figsize=(10, 5))
        plt.plot(np.mean(train_loss_l, axis=0))
        plt.plot(np.mean(val_loss_l, axis=0))
        plt.show()
    
    else:
        plt.figure(figsize=(30, 10))
        for i in range(len(train_loss_l)):
            plt.subplot(3, 5, i + 1)
            plt.plot(train_loss_l[i])
            plt.plot(val_loss_l[i])
        plt.show()
        
def plot_metrics(problem_name, problem, mean=False, metric="auc"):
    train_loss_l, val_loss_l, train_metric_l, val_metric_l = load_results(problem_name, problem, metric)
    if mean:
        plt.figure(figsize=(10, 5))
        plt.plot(np.mean(train_metric_l, axis=0))
        plt.plot(np.mean(val_metric_l, axis=0))
        plt.show()
    
    else:
        plt.figure(figsize=(30, 10))
        for i in range(len(train_loss_l)):
            plt.subplot(3, 5, i + 1)
            plt.plot(train_metric_l[i])
            plt.plot(val_metric_l[i])
            plt.ylim(0.0, 1.0)
        plt.show()