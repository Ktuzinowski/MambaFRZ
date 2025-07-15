import torch
from torch.utils.data import Dataset
import pickle

# Compressed Dataset Definition
class MambaFRZDataset(Dataset):
    def __init__(self, pt_file_loc, context_window_size):
        super().__init__()
        self.context_window_size = context_window_size
        
        self.data = pickle.load(
            open(pt_file_loc, "rb")
        )
        
        self.seed_list = list(self.data.keys())
        print(f"Seeds List: {self.seed_list}")
        
        self.num_seeds = len(self.seed_list)
        print(f"Number of Seeds: {self.num_seeds}")
        
        self.conv_list = list(self.data[self.seed_list[0]][0].keys())
        print(f"CNNs List: {self.conv_list}")
        
        self.num_convs = len(self.conv_list)
        print(f"Number of CNN layers: {self.num_convs}")
        
        self.num_epochs = len(self.data[self.seed_list[0]][1][self.conv_list[0]])
        print("Number of epochs", self.num_epochs)
    def __len__(self):
        return self.num_epochs * self.num_seeds * self.num_convs
    
    def __getitem__(self, idx):
        # print("IDX:", idx)
        seq_len_index = idx // (self.num_convs * self.num_seeds)
        conv_idx = idx % self.num_convs
        seed_idx = (idx % (self.num_convs * self.num_seeds)) // self.num_convs
        
        seed = self.seed_list[seed_idx]
        conv_layer = self.conv_list[conv_idx]
        label = self.data[seed][1][conv_layer][seq_len_index]
        input_weights = self.data[seed][0][conv_layer][0:self.context_window_size * seq_len_index + self.context_window_size]
        # print(f"Conv layer: {conv_layer}, seed {seed}, current idx {idx}")
        return input_weights, label