import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits import axes_grid1
from typing import List, Dict
from functools import partial
from torch.utils.data import Subset, DataLoader
from torch.utils.data import DataLoader
import tqdm
import gc

def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str = 'cuda:1'):
        """
        :param model1: (nn.Module) model 1
        :param model2: (nn.Module) model 2
        :param model1_name: (str) name of model 1
        :param model2_name: (str) name of model 2
        :param model1_layers: (List[str]) list of layers to compare for model 1
        :param model2_layers: (List[str]) list of layers to compare for model 2
        :param device: (str) device to run the model on
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        self.model1_layers = model1_layers
        self.model2_layers = model2_layers

        self._insert_hooks()

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):
        # Ensure activations are explicity on the CPU
        out = out.detach().cpu()
        if model == "model1":
            self.model1_features[name] = out
        elif model == "model2":
            self.model2_features[name] = out
        else:
            raise RuntimeError("Unknown model name for _log_layer")

    def _insert_hooks(self):
        self.hook_handles = []
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    handle = layer.register_forward_hook(partial(self._log_layer, "model1", name))
                    self.hook_handles.append(handle)
            else:
                self.model1_info['Layers'] += [name]
                handle = layer.register_forward_hook(partial(self._log_layer, "model1", name))
                self.hook_handles.append(handle)
        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    handle = layer.register_forward_hook(partial(self._log_layer, "model2", name))
                    self.hook_handles.append(handle)
            else:
                self.model2_info['Layers'] += [name]
                handle = layer.register_forward_hook(partial(self._log_layer, "model2", name))
                self.hook_handles.append(handle)

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()
    
    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None,
                num_times_iterate_over_test_dataset=1, percentage_of_batches=0.25) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """
        if dataloader2 is None:
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3, device="cpu")

        num_batches = min(len(dataloader1), len(dataloader2)) * num_times_iterate_over_test_dataset
        num_batches = int(num_batches * percentage_of_batches)
        print(f"Total number of batches in use: {num_batches}")

        smaller_total = num_batches

        for _ in range(num_times_iterate_over_test_dataset):
          for (x1, *_) in tqdm.tqdm(dataloader1, desc="| Comparing features |", total=smaller_total):
              if smaller_total == 0:
                break
              smaller_total -= 1
              self.model1_features = {}
              self.model2_features = {}
              with torch.no_grad():
                # Forward pass remains on GPU
                _ = self.model1(x1.to(self.device))
                _ = self.model2(x1.to(self.device))

                for i, (name1, feat1) in enumerate(self.model1_features.items()):
                    X = feat1.flatten(1).cpu() # Move features to CPU
                    # print(f"Shape of Activation X from layer {name1}: {X.shape}")
                    K = X @ X.t()
                    K.fill_diagonal_(0.0)
                    self.hsic_matrix[i, :, 0] += self._HSIC(K, K)

                    for j, (name2, feat2) in enumerate(self.model2_features.items()):
                        Y = feat2.flatten(1).cpu() # Move features to CPU
                        # print(f"Shape of Activation Y from layer {name2}: {Y.shape}")
                        L = Y @ Y.t()
                        L.fill_diagonal_(0.0)
                        assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                        self.hsic_matrix[i, j, 1] += self._HSIC(K, L)
                        self.hsic_matrix[i, j, 2] += self._HSIC(L, L)
              # Explicitly delete all intermediate tensors
              del X, K, Y, L
              torch.cuda.empty_cache()
              gc.collect()
        self.hsic_matrix[:, :, 0] /= num_batches
        self.hsic_matrix[:, :, 1] /= num_batches
        self.hsic_matrix[:, :, 2] /= num_batches
        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset']
        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()

    def __del__(self):
      # Removing handles
      for handle in self.hook_handles:
        handle.remove()
      self.hook_handles.clear()

      # Remove models safely
      if hasattr(self, "model1"):
        del self.model1
      if hasattr(self, "model2"):
        del self.model2

      # Remove feature storage
      self.model1_features.clear()
      self.model2_features.clear()

      # Remove layer information
      self.model1_layers = None
      self.model2_layers = None
      self.model1_info.clear()
      self.model2_info.clear()

      # Remove HSIC matrix
      if hasattr(self, "hsic_matrix"):
          del self.hsic_matrix

      # Run garbage collection and clear CUDA memory
      gc.collect()
      torch.cuda.empty_cache()