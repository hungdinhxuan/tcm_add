import numpy as np
import os
import sys
import torch
import importlib
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from typing import Dict, Optional, Union
from pathlib import Path

def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

def read_metadata(dir_meta, is_eval=False):
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()
    
    if (is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list

def read_metadata_pro(dir_meta):
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    for line in l_meta:
            key,label = line.strip().split()
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
    return d_meta,file_list

def read_metadata_eval(dir_meta, no_label=False):
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()
    
    for line in l_meta:
        if no_label:
            key= line.strip()
        else:
            key,label = line.strip().split()
        file_list.append(key)
    return file_list

def read_metadata_other(dir_meta, is_train=False, is_eval=False, is_dev=False):
    # bonafide: 1, spoof: 0
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()
    if (is_train):
        for line in l_meta:
            utt, subset, label = line.strip().split()
            if subset == 'train':
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0
    if (is_dev):
        for line in l_meta:
            utt, subset, label = line.strip().split()
            if subset == 'dev':
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0    
    if (is_eval):

        for line in l_meta:
            utt, subset, label = line.strip().split()
            if subset == 'eval':
                file_list.append(utt)
                d_meta[utt] = 1 if label == 'bonafide' else 0
    return file_list, d_meta

def reproducibility(random_seed, args=None):                                  
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn_deterministic = True
    cudnn_benchmark = False
    print("cudnn_deterministic set to False")
    print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return

def my_collate(batch): #Dataset return sample = (utterance, target, nameFile) #shape of utterance [1, lenAudio]
  data = [dp[0] for dp in batch]
  label = [dp[1] for dp in batch]
  nameFile = [dp[2] for dp in batch]
  return (data, label, nameFile) 


class ExperimentLogger:
    """
    A unified logging interface that supports both TensorBoard and Weights & Biases.
    This class handles logging of training metrics, model parameters, and other experiment data.
    """
    def __init__(self, 
                 experiment_name: str,
                 log_dir: str = 'runs',
                 use_tensorboard: bool = True,
                 use_wandb: bool = True,
                 wandb_project: Optional[str] = "multiview",
                 config: Optional[Dict] = None):
        """
        Initialize the experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to store TensorBoard logs
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: Name of the W&B project
            config: Configuration dictionary for the experiment
        """
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        # Initialize TensorBoard
        if use_tensorboard:
            self.tb_writer = SummaryWriter(
                log_dir=Path(log_dir) / experiment_name
            )
            
        # Initialize Weights & Biases
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=config
            )
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics to both TensorBoard and W&B if enabled.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current training step or epoch
        """
        # Log to TensorBoard
        if self.use_tensorboard:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)
                
        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_model(self, model: torch.nn.Module, epoch: int):
        """
        Log model architecture and gradients.
        
        Args:
            model: PyTorch model
            epoch: Current epoch number
        """
        if self.use_wandb:
            wandb.watch(model, log="gradients")
    
    def close(self):
        """Close all logging connections."""
        if self.use_tensorboard:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()