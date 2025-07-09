import torch
from torch.utils.data import Dataset
import pickle

# Compressed Dataset Definition
class CompressedFreezeDataset(Dataset):
  def __init__(self, pt_file_loc):
    super().__init__()
    with open(pt_file_loc, "rb") as f:
      data_dict = pickle.load(f)
    print(len(data_dict['data']))
    print(len(data_dict['labels']))
    self.data = torch.stack([entry[0] for entry in data_dict['data']], dim=0) # [N, ...]
    self.layer_names = [entry[1] for entry in data_dict['data']]
    self.epochs = [entry[2] for entry in data_dict['data']]
    self.seeds = [entry[3] for entry in data_dict['data']]
    self.labels = torch.tensor(data_dict['labels'], dtype=torch.long) # [N]
                
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    input = self.data[idx]
    output = self.labels[idx]

    return (input, self.layer_names[idx], self.epochs[idx], self.seeds[idx]), output