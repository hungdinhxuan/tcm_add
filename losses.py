import torch.nn as nn
import torch.nn.functional as F

def loss_fn_kd(outputs, teacher_outputs, alpha, T,reduction='batchmean'):
    """
    Compute the knowledge-distillation (KD) loss given outputs.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss(reduction=reduction)(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) 
    return KD_loss