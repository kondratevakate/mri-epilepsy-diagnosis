import time
from comet_ml import Experiment

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
from scipy import stats
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def run_one_epoch(model, loader, criterion, train, device,
                  optimizer=None, scheduler=None, experiment= False):
    
    model.to(device)
    model.train(train)

    losses = []
    probs = []
    targets = []

    for data, target, _ in tqdm(loader):
        data = data.to(device, dtype=torch.float)
        target = target.long().to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        
        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
                    
        losses.append(loss.data.cpu().numpy())
        
        probs.extend(F.softmax(outputs, dim=-1).cpu().data.numpy()[:, 1])
        targets.extend(list(target.cpu().data.numpy()))
                
        if experiment:
            if train:
                train_loss = loss.data.cpu().numpy()
                experiment.log_metric("train_loss", train_loss)
            elif train == False:
                val_loss = loss.data.cpu().numpy()
                experiment.log_metric("val_loss", val_loss)
        
        del data, target, outputs, loss

    return losses, probs, targets


def train(model, optimizer, scheduler, train_dataloader, val_dataloader, device,
          metric, verbose=0, model_save_path=None, max_epoch=20, 
          eps=3e-3, max_patience=10, experiment = False):
    
    criterion = nn.CrossEntropyLoss()

    patience = 0
    best_metric = 0

    epoch_train_loss, last_train_loss, epoch_train_metric, last_train_metric, = [], None, [], None
    epoch_val_loss, last_val_loss, epoch_val_metric, last_val_metric, = [], None, [], None

    for epoch in range(max_epoch):
        start_time = time.time()

        # 1. Train
        train_losses, train_probs, train_targets = run_one_epoch(
            model, train_dataloader, criterion, True, device, optimizer, scheduler, experiment)

        # 2. Inference
        if val_dataloader is not None:
            with torch.no_grad():
                val_losses, val_probs, val_targets = run_one_epoch(
                    model, val_dataloader, criterion, False, device, optimizer, scheduler, experiment)

        # 3. Metrics
        epoch_train_loss.append(np.mean(train_losses))
        epoch_train_metric.append(metric(train_targets, train_probs))
        if experiment:
            experiment.log_metrics({"mean_train_loss": np.mean(train_losses),
                                    "train_metric": metric(train_targets, train_probs)},
                                  epoch = epoch)
        if val_dataloader is not None:
            epoch_val_loss.append(np.mean(val_losses))
            epoch_val_metric.append(metric(val_targets, val_probs))
            if experiment:
                experiment.log_metrics({"mean_val_loss": np.mean(val_losses),
                                    "val_metric": metric(val_targets, val_probs)},
                                  epoch = epoch)

        # 4. Print metrics
        if verbose:
            clear_output(True)
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, max_epoch, time.time() - start_time))
            print("  training loss (in-iteration): \t{:.6f}".format(epoch_train_loss[-1]))
            if val_dataloader is not None:
                print("  validation loss: \t\t\t{:.6f}".format(epoch_val_loss[-1]))
            print("  training {}: \t\t\t{:.2f}".format(metric.__name__, epoch_train_metric[-1]))
            if val_dataloader is not None:
                print("  validation {}: \t\t\t{:.2f}".format(metric.__name__, epoch_val_metric[-1]))    
            
        # 5. Plot metrics
        if verbose:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            plt.figure(figsize=(10, 5))
            axes[0].plot(epoch_train_loss, label='train')
            if val_dataloader is not None:
                axes[0].plot(epoch_val_loss, label='val')
            axes[0].set_xlabel('epoch')
            axes[0].set_ylabel('loss')
            axes[0].legend()
            axes[1].plot(epoch_train_metric, label='train')
            if val_dataloader is not None:
                axes[1].plot(epoch_val_metric, label='val')
            axes[1].set_ylim([0, 1.05])
            axes[1].set_xlabel('epoch')
            axes[1].set_ylabel(metric.__name__)
            axes[1].legend()
            plt.show()
        
#         5. Early stopping, best metrics, save model
        if val_dataloader is not None and epoch_val_metric[-1] > best_metric:
            patience_ = 0
            best_metric = epoch_val_metric[-1]
            last_train_metric, last_val_metric = epoch_train_metric[-1], epoch_val_metric[-1]
            last_train_loss, last_val_loss = epoch_train_loss[-1], epoch_val_loss[-1]
            if model_save_path is not None:
                    torch.save(model.state_dict(), model_save_path)
                    
        elif val_dataloader is None and epoch_train_metric[-1] >= best_metric:
            patience = 0
            best_metric = epoch_train_metric[-1]
            last_train_metric = epoch_train_metric[-1]
            last_train_loss = epoch_train_loss[-1]
            if model_save_path is not None:
                torch.save(model.state_dict(), model_save_path)
        else:
            patience_ += 1

        if patience_ >= max_patience:
            print("Early stopping! Patience is out.")
            break
        if epoch_train_loss[-1] < eps:
            print("Early stopping! Train loss < eps.")
            break

        last_train_metric = epoch_train_metric[-1]
        last_train_loss = epoch_train_loss[-1]
    
#         if experiment:
#             experiment.log_epoch_end(epoch)
        if model_save_path is not None:
                torch.save(model.state_dict(), model_save_path)

    return last_train_loss, last_train_metric, last_val_loss, last_val_metric


def stratified_batch_indices(indices, labels):
    dominating_label = stats.mode(labels)[0][0]
    idx0 = indices[labels == dominating_label]
    idx1 = indices[labels != dominating_label]
    step = np.ceil(len(idx0) / len(idx1)) + 1
    assert step >= 1.
    result = []
    j0 = 0
    j1 = 0
    for i in range(len(indices)):
        if (i % step == 0 or j0 == len(idx0)) and j1 < len(idx1):
            result.append(idx1[j1])
            j1 += 1
        else:
            result.append(idx0[j0])
            j0 += 1
    result = np.array(result)
    assert len(result) == len(indices)
    return result

def cross_val_score(cnn_model, train_dataset, cv, device, metric, holdout_idx = None,
                    model_load_path=None, batch_size=10, val_dataset=None, transfer=False, 
                    finetune=False, experiment = False, max_epoch=20):
    
    assert not (transfer and finetune)
    assert (transfer == False) or (transfer == True and model_load_path is not None)

    use_rest = True
    if val_dataset is None:  # smri case or fmri case without rest
        val_dataset = train_dataset
        use_rest = False
    if holdout_idx is not None:
        cv_splits = list(cv.split(X=np.arange(len(holdout_idx)), y=train_dataset.target[holdout_idx.index]))
    else:
        cv_splits = list(cv.split(X=np.arange(len(train_dataset)), y=train_dataset.target))
    
    val_metrics = []

    for i in range(len(cv_splits)):
        train_idx, val_idx = cv_splits[i]

        # train data
        if model_load_path is None or transfer or finetune:
            train_idx = stratified_batch_indices(train_idx, train_dataset.target[train_idx])
            train_loader = DataLoader(Subset(train_dataset, train_idx),
                                      shuffle=False,
                                      batch_size=batch_size,
                                      drop_last=False)

        # val data
        if use_rest:
            val_mask = (np.isin(val_dataset.pids, train_dataset.pids[train_idx]) == False)
            val_idx = np.arange(len(val_dataset))[val_mask]
            del val_mask
        
        val_loader = DataLoader(Subset(val_dataset, val_idx),
                                shuffle=False,
                                batch_size=batch_size,
                                drop_last=False)

        eps = 1e-2 if use_rest else 3e-3
        model, optimizer, scheduler = create_model_opt(cnn_model)
        if model_load_path is None or transfer or finetune:
            _, _, _, last_val_metric = train(model, optimizer, scheduler,
                                         train_loader, val_loader, device,
                                        metric=metric, 
                                        verbose=1,  
                                        max_epoch=max_epoch,
                                        eps=eps,
                                        experiment=experiment,
                                        model_save_path = 'checkpoint_1.pth')
            val_metrics.append(last_val_metric)
                
        if model_load_path:  # no train, just validation
            model, optimizer, scheduler = create_model_opt(cnn_model, model_load_path = model_load_path, transfer=False)
            criterion = nn.CrossEntropyLoss()
            with torch.no_grad():
                val_losses, val_probs, val_targets = run_one_epoch(
                    model, val_loader, criterion, False, device, scheduler)
                if metric.__name__ == 'roc_auc_score':
                    val_metric = metric(val_targets, val_probs)
                if metric.__name__ == 'accuracy_score':
                    val_metric = metric(val_targets, np.where(np.array(val_probs)<=0.5, 0, 1))
            
            val_metrics.append(val_metric)
        if model_load_path is None:
            del train_loader
        del  val_loader, model, optimizer, scheduler

    return val_metrics

def create_model_opt(model, model_load_path=None, 
                     input_shape = (192,192,192), n_fc_units = 192, transfer=False, lr=1e-5, patience = 2):
    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))  
    if transfer:
        for param in model.parameters():
            param.requires_grad = False
        
        last = torch.nn.Linear(128, 2)
        modules = list(list(model.children())[0].children())[:-1] + [last]
        model = torch.nn.Sequential(torch.nn.Sequential(*modules))
        
        opt = torch.optim.Adam(last.parameters(), lr, weight_decay = 0.01)
#         scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=patience, threshold=0.001)
    else:
        opt = torch.optim.Adam(model.parameters(), lr,weight_decay = 0.01)
#         scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=patience, threshold=0.001)
        
    return model, opt, scheduler