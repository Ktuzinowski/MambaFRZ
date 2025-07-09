import torch
import torch.nn as nn
import random
from torch.optim.lr_scheduler import _LRScheduler

def random_sample(input_tensor, size):
  rand_num = set()
  input_tensor = input_tensor.reshape(-1)
  while 1:
    rand_num.add(random.randint(0, len(input_tensor)-1))
    if len(rand_num) >= size:
      break
  result = []
  rand_num = sorted(rand_num)
  for i in rand_num:
      result.append(input_tensor[i].item())
  result = torch.tensor(result).unsqueeze(0)
  return result

def is_cnn_layer(module):
  # Check if the module has no children
  has_no_children = len(list(module.children())) == 0

  # Check if the module is a Conv2d
  is_target_type = isinstance(module, (nn.Conv2d))

  return has_no_children and is_target_type

def is_bn_layer(module):
  # Check if the module has no children
  has_no_children = len(list(module.children())) == 0

  # Check if the module is a BatchNorm2d layer
  is_target_type = isinstance(module, (nn.BatchNorm2d))

  return has_no_children and is_target_type

def soft_cross_entropy(outputs, soft_targets):
  log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
  return -torch.mean(torch.sum(soft_targets * log_probs, dim=1))

class WarmUpLR(_LRScheduler):
  """warmup_training learning rate scheduler
     Args:
       optimizer: optimzier(e.g. SGD)
       total_iters: total iterations of warmup phase
  """
  def __init__(self, optimizer, total_iters, last_epoch=-1):
    self.total_iters = total_iters
    super().__init__(optimizer, last_epoch)

  def get_lr(self):
    """we will use the first m batches, and set the learning
       rate to base_lr * m / total_iters
    """
    return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]