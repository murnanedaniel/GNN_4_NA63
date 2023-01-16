import sys, os
import logging
import time
import warnings
import itertools
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .utils import load_processed_datasets

class GNNBase(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        
        """
        Initialise the Lightning Module that can scan over different Equivariant GNN training regimes
        """
        # Assign hyperparameters
        self.save_hyperparameters(hparams)

        if "graph_construction" not in self.hparams: self.hparams["graph_construction"] = None
        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage="fit"):

        data_split = self.hparams["data_split"].copy()

        if stage == "fit":
            data_split[2] = 0 # No test set in training
        elif stage == "test":
            data_split[0], data_split[1] = 0, 0 # No train or val set in testing

        if (self.trainset is None) and (self.valset is None) and (self.testset is None):
            self.load_data(stage, self.hparams["input_dir"], data_split)
        
        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")
            self.logger.experiment.define_metric("acc" , summary="max")
            self.logger.experiment.define_metric("inv_eps" , summary="max")
        except Exception:
            warnings.warn("Failed to define figures of merit, due to logger unavailable")
            
        print(time.ctime())

    def load_data(self, stage, input_dir, data_split):
        """
        Load in the data for training, validation and testing.
        """

        # if stage == "fit":
        for data_name, data_num in zip(["trainset", "valset", "testset"], data_split):
            print(f"Loading {data_name} with {data_num} files")
            if data_num > 0:
                dataset = GraphDataset(input_dir, data_name, data_num, stage, self.hparams)
                setattr(self, data_name, dataset)
        
    def concat_feature_set(self, batch):
        """
        Useful in all models to use all available features of size == len(x)
        """
        
        all_features = []
        for feature in self.hparams["feature_set"]:
            if len(batch[feature]) == len(batch.x):
                all_features.append(batch[feature])
            else:
                all_features.append(batch[feature][batch.batch])
        return torch.stack(all_features).T

    def get_metrics(self, targets, output):
        
        prediction = torch.sigmoid(output)
        tp = (prediction.round() == targets).sum().item()
        acc = tp / len(targets)
        
        try:
            auc = roc_auc_score(targets.bool().cpu().detach(), prediction.cpu().detach())
        except Exception:
            auc = 0
        fpr, tpr, _ = roc_curve(targets.bool().cpu().detach(), prediction.cpu().detach())
        
        # Calculate which threshold gives the best signal goal
        signal_goal_idx = abs(tpr - self.hparams["signal_goal"]).argmin()
        
        eps_fpr = fpr[signal_goal_idx]
        eps_tpr = tpr[signal_goal_idx]

        
        return acc, auc, eps_fpr, eps_tpr
    
    def define_truth(self, batch):
        """
        For now, define truth as:
            - Fake if y == 0; that is, there is no trident event
            - True if y > 0; that is, there is a trident event but we don't care what type it is

        TODO: Handle different types of trident events
        """

        truth = batch.y > 0

        return truth

    def apply_loss_function(self, output, batch):
        """
        For now, average over all fake and true samples.

        TODO: Get positive and negative loss separately, then sum together to ensure a balanced loss
        """

        truth = self.define_truth(batch)

        # Get positive and negative loss separately
        pos_loss = F.binary_cross_entropy_with_logits(output[truth], truth[truth].float())
        neg_loss = F.binary_cross_entropy_with_logits(output[~truth], truth[~truth].float())

        # Mean together to ensure a balanced loss
        loss = (self.hparams["pos_weight"]*pos_loss + neg_loss)/2

        return loss

    def training_step(self, batch, batch_idx):
                
        output = self(batch).squeeze(-1)

        loss = self.apply_loss_function(output, batch)
        
        acc, auc, eps, eps_eff = self.get_metrics(batch.y.bool(), output)
        
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=False, on_epoch=True)

        return loss        

    def shared_val_step(self, batch):

        output = self(batch).squeeze(-1)
        loss = self.apply_loss_function(output, batch)

        acc, auc, eps, eps_eff = self.get_metrics(self.define_truth(batch), output)
        
        try:
            current_lr = self.optimizers().param_groups[0]["lr"]
        except Exception:
            current_lr = 0
        
        self.log_dict({"val_loss": loss, "current_lr": current_lr}, on_step=False, on_epoch=True)
        
        return {
            "loss": loss,
            "outputs": output,
            "targets": batch.y,
            "acc": acc,
            "auc": auc,
            "eps": eps,
            "eps_eff": eps_eff,
        }

    def validation_step(self, batch, batch_idx):
        return self.shared_val_step(batch)

    def test_step(self, batch, batch_idx):
        return self.shared_val_step(batch)
        
    def shared_end_step(self, step_outputs):
        # Concatenate all predictions and targets
        preds = torch.cat([output["outputs"] for output in step_outputs])
        targets = torch.cat([output["targets"] for output in step_outputs])

        # Calculate the ROC curve
        acc, auc, eps, eps_eff = self.get_metrics(targets, preds)

        if eps != 0:
            self.log_dict({"acc": acc, "auc": auc, "inv_eps": 1/eps, "eps_eff": eps_eff})
    
    def validation_epoch_end(self, step_outputs):
        self.shared_end_step(step_outputs)

    def test_epoch_end(self, step_outputs):
        self.shared_end_step(step_outputs)

        
    
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=self.hparams["train_batch"], num_workers=10, shuffle=False)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=self.hparams["val_batch"], num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0001,
                amsgrad=True,
            )
        ]
        if "scheduler" not in self.hparams or self.hparams["scheduler"] is None or self.hparams["scheduler"] == "StepLR":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.StepLR(
                        optimizer[0],
                        step_size=self.hparams["patience"],
                        gamma=self.hparams["factor"],
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            ]
        elif self.hparams["scheduler"] == "CosineWarmLR":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer[0],
                        T_0 = self.hparams["patience"], 
                        T_mult=2,
                    ),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
        return optimizer, scheduler
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class GraphDataset(Dataset):
    """
    The custom default GNN dataset to load graphs off the disk
    """

    def __init__(self, input_dir, data_name = "trainset", num_events = 0, stage="fit", hparams=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(input_dir, transform, pre_transform, pre_filter)
        
        self.input_dir = input_dir
        self.data_name = data_name
        self.hparams = hparams
        self.num_events = num_events
        self.stage = stage

        self.load_files()

    def load_files(self):
        self.input_paths = sorted(os.listdir(os.path.join(self.input_dir, self.data_name)))
        num_files = (self.num_events // 100000) + 1 if self.num_events > 0 else 0
        self.input_paths = [os.path.join(self.input_dir, self.data_name, path) for path in self.input_paths[:num_files]]
        
        self.loaded_files = [torch.load(file) for file in tqdm(self.input_paths)]
        self.loaded_files = list(itertools.chain.from_iterable(self.loaded_files))[:self.num_events]
        
        
    def len(self):
        return len(self.loaded_files)

    def get(self, idx):

        event = self.loaded_files[idx]
        return self.preprocess_event(event)

    

    # Add concatenation method logic
    


    def preprocess_event(self, event):
        """
        Process event before it is used in training and validation loops
        """

        self.handle_edge_list(event)
        self.scale_features(event)

        return event

    def handle_edge_list(self, event):
        """ 
        For now, make graph fully connected.
        TODO: Add this as a configurable choice once trying gravnet
        """

        num_nodes = event.x.shape[0]
        # Get a list of all possible edges, using torch combinations
        edge_list = torch.combinations(torch.arange(num_nodes), r=2)
        edge_list = torch.cat([edge_list, edge_list.flip(1)], dim=0)
        # Add the edges to the event
        event.edge_index = edge_list.t().contiguous()

    def scale_features(self, event):
        """
        Scale up x and y by 1000 to be on same scale as z
        """

        event.x[:,0] *= 1000
        event.x[:,1] *= 1000

    def unscale_features(self, event):
        """
        Scale down x and y by 1000 to return to the original scale. Used in prediction for saving the inferred event
        """

        event.x[:,0] /= 1000
        event.x[:,1] /= 1000
