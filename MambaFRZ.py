import torch
import torch.nn as nn
from mamba_ssm import Mamba2

class Mamba2Block(nn.Module):
  def __init__(self, feature_dim, ssm_state_expansion_factor, d_conv):
    super(Mamba2Block, self).__init__()
    self.mamba2 = Mamba2(
        d_model=feature_dim,  # Should match 1024, feature dimension of each token being 1024 sampled weights
        d_state=ssm_state_expansion_factor,     # State-space size, set to arbitrary number until verified empirically
        d_conv=d_conv,   # Convolution projection dimension, set to 2 since mixing of weights doesn't make sense but needs to be 2 at minimum
        expand=2              # Expansion factor, selected as 2 because of ablation study produced best perplexity
    )

  def forward(self, x):
    # Expecting input of shape (batch_size, seq_len=30, feature_dim=1024)
    return self.mamba2(x)


class MLP(nn.Module):
  def __init__(self, num_i, num_h, num_o):
    super(MLP, self).__init__()
    self.linear1 = nn.Linear(num_i, num_h)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(num_h, num_h)
    self.relu2 = nn.ReLU()
    self.linear3 = nn.Linear(num_h, num_o)
    self.dropout = nn.Dropout(0.1)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear2(x)
    x = self.relu2(x)
    x = self.dropout(x)
    x = self.linear3(x)
    return x

class Mamba2AttentionModule(nn.Module):
  def __init__(self, feature_dim, ssm_state_expansion_factor, mlp_hid_channel, mlp_out_channel, d_conv=2):
    super(Mamba2AttentionModule, self).__init__()
    self.mamba_block = Mamba2Block(feature_dim, ssm_state_expansion_factor, d_conv)
    self.norm = nn.LayerNorm(feature_dim)
    self.output_mlp = MLP(feature_dim, mlp_hid_channel, mlp_out_channel)
    self.dropout = nn.Dropout(p=0.2)

  def forward(self, x):
    # Ensure input is (batch_size, seq_len=30, feature_dim=1024)
    shortcut = x
    x = self.norm(x) # Pre-layer normalization
    x = self.mamba_block(x)  # Process sequence with Mamba
    x = self.dropout(x)
    x = x + shortcut
    x = x.mean(dim=1)
    output = self.output_mlp(x)
    # output = self.output_mlp(x[:, -1, :])  # Use last time step for prediction
    return output

def initialize_mamba2_predictor(feature_dim, ssm_state_expansion_factor, mlp_hid_channel, mlp_out_channel, d_conv=2):
  return Mamba2AttentionModule(feature_dim, ssm_state_expansion_factor, mlp_hid_channel, mlp_out_channel, d_conv)