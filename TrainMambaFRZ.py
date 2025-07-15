import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import re
from MambaFRZ import initialize_mamba2_predictor
from MambaFRZDataset import MambaFRZDataset
from collections import defaultdict
        
def main(args):
  root_dir = f"{args.name_of_experiment}/context_window_{args.context_window_size}"
  total_count = args.number_of_samples
  counter = 0
  name_of_experiment = args.name_of_experiment
  window_size = args.context_window_size
    
  train_dataset = MambaFRZDataset(f"{root_dir}/mambafrz_training_dataset.pkl", args.context_window_size)
  print("Length of Dataset:", len(train_dataset))
  testing_dataset = MambaFRZDataset(f"{root_dir}/mambafrz_testing_dataset.pkl", args.context_window_size)
  print("Length of Dataset:", len(testing_dataset))

  # REPRODUCIBILITY WITH RANDOM SEED
  random.seed(1234)
    
  batch_size = 8
  num_workers = 0
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  print(f"Number of batches in Train Loader: {len(train_loader)}")
  testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  print(f"Number of batches in Testing Loader: {len(testing_loader)}")
  
  feature_dim = args.re_size
  mlp_hid_channel = 256
  mlp_out_channel = 2
  ssm_state_expansion_factor = 16
  predictor = initialize_mamba2_predictor(feature_dim=feature_dim, ssm_state_expansion_factor=ssm_state_expansion_factor, mlp_hid_channel=mlp_hid_channel, mlp_out_channel=mlp_out_channel)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  print(torch.cuda.is_available())
  predictor.to(device)

#   label_smoothing = 0.2 # included label smoothing

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(device)

  optimizer = optim.AdamW(predictor.parameters(),
                        lr=1e-4,    # try 1e-3 then 1e-4
                        weight_decay=1e-5)  # small L2 to regularize
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True)
  
  num_epochs = args.num_epochs
  
  model_save_path = f"{root_dir}/checkpoints"
  os.makedirs(model_save_path, exist_ok=True)
  
  frozen_count = 0
  non_frozen_count = 0
  best_training_acc = 0.0
  
  training_epoch_loss = []
  for epoch in range(num_epochs):
    predictor.train()
    train_running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = predictor(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
      optimizer.step()
      train_running_loss += loss.item()
      correct += sum([torch.argmax(pred).item() == label.item() for pred, label in zip(outputs, labels)])
      total += labels.size(0)
    train_running_loss /= len(train_loader)
    training_epoch_loss.append(train_running_loss)
    epoch_accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_running_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")
    
    correct = 0
    testing_running_loss = 0.0
    total = 0
    for inputs, labels in testing_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = predictor(inputs)
        loss = criterion(outputs, labels)
        testing_running_loss += loss.item()
        correct += sum([torch.argmax(pred).item() == label.item() for pred, label in zip(outputs, labels)])
        total += labels.size(0)
    testing_running_loss /= len(testing_loader)
    epoch_testing_accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Testing Loss: {testing_running_loss:.4f}, Testing Accuracy: {epoch_testing_accuracy:.4f}")
    
    if epoch_testing_accuracy > best_training_acc:
      best_training_acc = epoch_testing_accuracy
      torch.save(predictor.state_dict(), os.path.join(model_save_path, f"mambafrz_trained_{epoch}.pth"))
      print(f"New Best Acc: {best_training_acc}")
  plt.plot(training_epoch_loss, label='Training Loss')
  plt.legend()
  plt.show()
    
    
      
class Args:
  def __init__(self, name_of_experiment, context_window_size, number_of_samples, re_size=1024, num_epochs=2):
    self.context_window_size = context_window_size
    self.name_of_experiment = name_of_experiment
    self.number_of_samples = number_of_samples
    self.re_size = re_size
    self.num_epochs = num_epochs

args = Args(name_of_experiment="mambafrz_20_conv_seed_25_experiment/longer_context_training_plus_testing_dataset", context_window_size=30, number_of_samples=64000, re_size=1024, num_epochs=10)
main(args)