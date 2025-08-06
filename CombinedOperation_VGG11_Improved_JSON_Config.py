import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg11, vgg11_bn
import pickle
import numpy as np
import random
import gc
import math
import re
from torchvision.transforms import v2
from cka import CKA
from models.utils import random_sample, is_cnn_layer, is_bn_layer, soft_cross_entropy, WarmUpLR
from models.datasets import get_dataloaders_for_dataset
from frz_predictor import initialize_mamba2_predictor, initialize_smartfrz_predictor
from collections import defaultdict
import json
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # needed for full determinism in some CUDA ops

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--configuration_file",
        type=str,
        default="config.json",
        help="Configuration file to use when computing CKA"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed to ensure reproducibility for dataset generation"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to compute CKA with"
    ) 
    
    args = parser.parse_args()
    return args
    
def main(config):
  best_acc = 0.0
  seed = config["seed"]
  name_of_experiment = config["name_of_experiment"]
  frz_predictor_type = config["frz_predictor_type"]
  save_fully_trained_ref_model = config["save_fully_trained_ref_model"]
  use_linear_restriction_for_layer_freezing = config["use_linear_restriction_for_layer_freezing"]
  use_predictions_from_frz_predictor_to_frz_layers = config["use_predictions_from_frz_predictor_to_frz_layers"]
  model_name = config["model_name"]
  dataset_name = config["dataset_name"]

  use_post_processing_window_for_frz_predictor = config["use_post_processing_window_for_frz_predictor"]
  post_processing_window_size = config["post_processing_window_size"]
  post_processing_percentage_of_window_in_order_to_frz = config["post_processing_percentage_of_window_in_order_to_frz"]
    
  threshold_for_post_processing_frz_decisions_in_window = int(post_processing_window_size * post_processing_percentage_of_window_in_order_to_frz)
  if threshold_for_post_processing_frz_decisions_in_window == 0:
    raise ValueError(f"Invalid Post-Processing Percentage for Freezing: {post_processing_percentage_of_window_in_order_to_frz}, reconsider with higher percentage")

  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # Create generator for DataLoader
  g = torch.Generator()
  g.manual_seed(seed)

  os.makedirs(f"{name_of_experiment}", exist_ok=True)
  # The location of the metrics for the experiment
  os.makedirs(f"{name_of_experiment}/seed_{seed}", exist_ok=True)
    
  device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

  context_window_size = config["context_window_size"]
  key_list = list()
  # The active layers
  conv_active = list()
  bn_active = list()

  # Store the layers
  conv_layer = list()
  bn_layer = list()

  # Store the frozen parameters
  conv_layer_param = dict()
  bn_layer_param = dict()

  # Store the weights for freezing prediction
  conv_active_weights = dict()

  # Store the frozen layers
  conv_frozen = list()
  bn_frozen = list()

  frz_predictor_config = config["frz_predictor_config"]

  if frz_predictor_type == "smartfrz":
    in_channel = frz_predictor_config["in_channel"]
    hid_channel = frz_predictor_config["hid_channel"]
    out_channel = frz_predictor_config["out_channel"]
    
    predictor = initialize_smartfrz_predictor(in_channel, hid_channel, out_channel)
  elif frz_predictor_type == "mambafrz":
    feature_dim = frz_predictor_config["feature_dim"]
    mlp_hid_channel = frz_predictor_config["mlp_hid_channel"]
    mlp_out_channel = frz_predictor_config["mlp_out_channel"]
    ssm_state_expansion_factor = frz_predictor_config["ssm_state_expansion_factor"]
    projected_dim = frz_predictor_config["projected_dim"]
    
    predictor = initialize_mamba2_predictor(feature_dim=feature_dim, projected_dim=projected_dim, ssm_state_expansion_factor=ssm_state_expansion_factor, mlp_hid_channel=mlp_hid_channel, mlp_out_channel=mlp_out_channel)

  if config["use_pretrained_frz_predictor"]:
    predictor.load_state_dict(torch.load(config["use_pretrained_frz_predictor_path"], map_location=device))
    predictor = predictor.to(device)

  num_classes = 100
  model = vgg11_bn(weights=None)
  model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
  # slimmed-down head
  classifier = nn.Sequential(
     nn.Linear(512, 256),
     nn.ReLU(inplace=True),
     nn.Dropout(0.5),
     nn.Linear(256, num_classes)
  )
  model.classifier = classifier
  net = model
  net.to(device)
    
  for name, layer in net.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
      conv_layer.append(layer)
    elif isinstance(layer, torch.nn.BatchNorm2d):
      bn_layer.append(layer)
    
  key = 0
  # Track which convolutional layer has been frozen to measure TFLOPs
  track_conv_frozen = {}

  if use_post_processing_window_for_frz_predictor:
    frz_decisions_per_layer = defaultdict(list)

  for name, layer in net.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
      key_list.append(key)
      conv_active_weights.setdefault(key, list())
      track_conv_frozen.setdefault(key, [name, []])
      conv_active.append(key)
      bn_active.append(key)
      key += 1
  
  if not save_fully_trained_ref_model:
      fully_trained_model = vgg11_bn(weights=None)
      fully_trained_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
      # slimmed-down head
      classifier = nn.Sequential(
          nn.Linear(512, 256),
          nn.ReLU(inplace=True),
          nn.Dropout(0.5),
          nn.Linear(256, num_classes)
      )
      fully_trained_model.classifier = classifier
      fully_trained_model = fully_trained_model.to(device)
      torch_state = torch.load(config["fully_trained_ref_model_path"], map_location=device)
      fully_trained_model.load_state_dict(torch_state)
      fully_trained_reference_model = fully_trained_model

  criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

  def seed_worker(worker_id):
    """
    Ensure reproducibility for each worker in DataLoader.
    """
    worker_seed = config["seed"] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

  num_workers = config["num_workers"]
  batch_size = config["batch_size"]
  cka_batch_size = config["cka_batch_size"]
  num_epochs = config["num_epochs"]
  warmup_epochs = config["warmup_epochs"]

  train_loader, test_loader, cka_loader = get_dataloaders_for_dataset(dataset_name, batch_size, cka_batch_size, num_workers, worker_init_fn=seed_worker, generator=g)

  optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
  train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs - warmup_epochs, eta_min=config["scheduler_eta_min"]) # learning rate decay
  iter_per_epoch = len(train_loader)
  warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epochs)

  tracker_of_layers_randomly_sampled_input_weights = {}
  tracker_of_cka_values_across_epochs = {}
  cka_freeze_layer_decisions = {}
  cka_freeze_layer_configuration = {}
  tracker_of_cka_window_values_across_epochs = {}
    
  set_of_all_layer_names = set()
  counter_for_cnn_layers = config["number_of_cnn_layers_for_cka"]
    
  sampled_indices = np.linspace(0, key - 1, counter_for_cnn_layers).astype(int)
  current_index_for_indices = 0
  index_counter_for_cnn_layers = 0
  
  for name, module in net.named_modules():
    if counter_for_cnn_layers == 0:
      break
    if is_cnn_layer(module):
      if index_counter_for_cnn_layers == sampled_indices[current_index_for_indices]:
        tracker_of_layers_randomly_sampled_input_weights[name] = []
        tracker_of_cka_values_across_epochs[name] = []
        cka_freeze_layer_decisions[name] = []
        cka_freeze_layer_configuration[name] = -1
        tracker_of_cka_window_values_across_epochs[name] = []
        
        set_of_all_layer_names.add(name)
    
        counter_for_cnn_layers -= 1
        current_index_for_indices += 1
      # Always increment index counter
      index_counter_for_cnn_layers += 1
  print(f"Set of all layers we are computing CKA for: {set_of_all_layer_names}")
  
  accuracy_list = []
  training_loss_list = []
  testing_loss_list = []

  cutmix = v2.CutMix(num_classes=num_classes)
  mixup = v2.MixUp(num_classes=num_classes)
  cutmix_or_mixup = transforms.RandomChoice([cutmix, mixup])

  for epoch in range(num_epochs):
    if epoch >= warmup_epochs:
      train_scheduler.step()

    net.train()
    train_running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      if len(conv_active):
        num_training_samples = len(train_loader.dataset)
        if i % int((num_training_samples/batch_size) / context_window_size) == 0:
          # Get the weights of current models, only for active layers
          for layer_index in conv_active:
            sampled_weights = random_sample(conv_layer[layer_index].weight.clone().detach().cpu().reshape(-1).unsqueeze(0), config["layer_weight_sample_size"])
            conv_active_weights[layer_index].append(sampled_weights)
            
            layer_name = track_conv_frozen[layer_index][0]
            if layer_name in set_of_all_layer_names:
                tracker_of_layers_randomly_sampled_input_weights[layer_name].append(sampled_weights.cpu())

      inputs, labels = data[0].to(device), data[1].to(device)
      inputs, labels = cutmix_or_mixup(inputs, labels)

      optimizer.zero_grad()

      outputs = net(inputs)
      loss = soft_cross_entropy(outputs, labels)
      loss.backward()
      optimizer.step()

      train_running_loss += loss.item()

      if epoch < warmup_epochs:
        warmup_scheduler.step()

    train_running_loss /= len(train_loader)
    if not save_fully_trained_ref_model:
        file_path = f"{name_of_experiment}/smartfrz_input_data/seed_{seed}_{config['device']}/epoch_{epoch}.pkl"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
          pickle.dump(tracker_of_layers_randomly_sampled_input_weights, file)

        del tracker_of_layers_randomly_sampled_input_weights
        torch.cuda.empty_cache()

        # Reset the tracker
        tracker_of_layers_randomly_sampled_input_weights = {}
        model_layers = []
        for name, module in net.named_modules():
          if name in set_of_all_layer_names:
            tracker_of_layers_randomly_sampled_input_weights[name] = []
            model_layers.append(name)

        is_previous_layer_frozen = True
        for model_layer in model_layers:
          if config["perform_similarity_guided_training"]:
            # Already froze layer, no need to compute CKA
            if cka_freeze_layer_configuration[model_layer] != -1:
                continue
          print(f"Comparing for {model_layer} at epoch {epoch}")
          cka = CKA(net, fully_trained_reference_model, model1_name=f'VGG11_{epoch}_Epoch', model2_name='VGG11_Fully_Trained', model1_layers=[model_layer], model2_layers=[model_layer], device=device)
          with torch.no_grad():
            cka.compare(cka_loader, num_times_iterate_over_test_dataset=config["num_times_to_iterate_over_dataset_for_cka"], percentage_of_batches=config["percentage_of_batches_from_testing_dataset_for_cka"])

          output_cka_dict = cka.export()
          tracker_of_cka_values_across_epochs[model_layer].append(output_cka_dict['CKA'][0].item())
          tracker_of_cka_window_values_across_epochs[model_layer].append(output_cka_dict['CKA'][0].item())
        
          moving_window_size = config["moving_window_size_similarity_guided_training"]
          stride_epochs = config["stride_epochs_similarity_guided_training"]
          variance_threshold = config["variance_threshold_similarity_guided_training"]
          cka_similarity_cutoff_value = config["cka_similarity_cutoff_value_similarity_guided_training"]

          if len(tracker_of_cka_window_values_across_epochs[model_layer]) > moving_window_size:
            # Remove previous CKA value if we have over the window size
            tracker_of_cka_window_values_across_epochs[model_layer] = tracker_of_cka_window_values_across_epochs[model_layer][stride_epochs:]

          if len(tracker_of_cka_window_values_across_epochs[model_layer]) == moving_window_size:
            # Begin checking variance threshold if we reached that point, use sample variance for unbiased estimate
            window_variance = np.var(tracker_of_cka_window_values_across_epochs[model_layer], ddof=1)

            current_cka_value = output_cka_dict['CKA'][0].item()
            if (window_variance < variance_threshold) and current_cka_value > cka_similarity_cutoff_value:
              cka_freeze_layer_decisions[model_layer].append(1)
              if is_previous_layer_frozen:
                cka_freeze_layer_configuration[model_layer] = epoch
                print(f"Similarity-Guided: Froze {model_layer}, Epoch {epoch}")

              if is_previous_layer_frozen and config["perform_similarity_guided_training"]:  
                # Signal to help freeze the corresponding batch normalization layer
                froze_cnn = False
                for name, module in net.named_modules():
                  if is_cnn_layer(module) and name == model_layer:
                    froze_cnn = True
                    module.weight.requires_grad = False
                    if hasattr(module, 'bias') and module.bias is not None:
                      module.bias.requires_grad = False
                  elif is_bn_layer(module) and froze_cnn:
                    print("FROZE CORRESPONDING BN LAYER!")
                    module.weight.requires_grad = False
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.requires_grad = False
                    break
            else:
              is_previous_layer_frozen = False
              cka_freeze_layer_decisions[model_layer].append(0) # 0 equals non-frozen layer
          else:
            is_previous_layer_frozen = False
            cka_freeze_layer_decisions[model_layer].append(0)

          # Explicitly call __del__ before deleting
          cka.__del__()
          del cka
          del output_cka_dict
          gc.collect() # Force garbage collection
          torch.cuda.empty_cache() # Free cached memory

    net.eval()
    totals = 0
    correct = 0
    testing_loss = 0
    with torch.no_grad():
      for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)

        totals += labels.size(0)
        correct += (predicted == labels).sum().item()
        testing_loss += criterion(outputs, labels).item()

    testing_loss /= len(test_loader)
    accuracy = correct / totals
    print("Epoch:", epoch, "Accuracy:", accuracy, "Training Loss:", train_running_loss, "Testing Loss:", testing_loss)

    # Save metrics
    testing_loss_list.append(testing_loss)
    accuracy_list.append(accuracy)
    training_loss_list.append(train_running_loss)
                                                                                                  
    # At the end of each epoch, make freeze predictions and record them
    # not going to freeze right now, but later on I will
    conv_freeze_list = []
    for p_index, p in enumerate(conv_active):
      if frz_predictor_type == "smartfrz":
        freeze_input = conv_active_weights[p][0]
        for index, weights in enumerate(conv_active_weights[p]):
            if index == 0:
                continue
            freeze_input = torch.cat((freeze_input, weights), 1)
            if index >= context_window_size - 1:
                break
        freeze_input = freeze_input.to(device)
        # Predict the freezing decision
        pred = predictor(freeze_input)
      elif frz_predictor_type == "mambafrz":
        freeze_input = conv_active_weights[p][0]
        for index, weights in enumerate(conv_active_weights[p]):
            if index == 0:
                continue
            freeze_input = torch.cat((freeze_input, weights), 0)
            if index >= context_window_size - 1:
              break
        freeze_input = freeze_input.unsqueeze(0).to(device)
        # Predict the freezing decision
        pred = predictor(freeze_input)
      print(f"Logits Frz Predictor, Conv# {p}: {pred}")
      prediction_for_freezing = torch.argmax(pred).item()
      if prediction_for_freezing == 1:
        if use_post_processing_window_for_frz_predictor:
            frz_decisions_per_layer[p].append(1)
            print(f"Post-Processing: Layer Decisions Conv# {p} {frz_decisions_per_layer[p]}")
        else:
            conv_freeze_list.append(p)
            print(f"Conv# {p}, Layer {track_conv_frozen[p][0]}, Frz Epoch {epoch}")

        track_conv_frozen[p][1].append(1)
      else:
        track_conv_frozen[p][1].append(0)
        if use_post_processing_window_for_frz_predictor:
            frz_decisions_per_layer[p].append(0)
            print(f"Post-Processing: Layer Decisions for Conv# {p} {frz_decisions_per_layer[p]}")
        if use_linear_restriction_for_layer_freezing:
            if use_post_processing_window_for_frz_predictor:
                # Do not skip making predictions on layers if we are using a post-processing window
                continue
            # Make all subsequent p indices append a 0
            for extended_p_index, extended_p in enumerate(conv_active):
                if extended_p_index <= p_index:
                    continue
                else:
                    track_conv_frozen[extended_p][1].append(0)
            break
    
    # After each epoch, we need to delete the weights contained previously
    current_keys = set(conv_active_weights.keys())
    del conv_active_weights
    conv_active_weights = dict()
    key = 0
    for name, layer in net.named_modules():
      if isinstance(layer, torch.nn.Conv2d):
        if key in current_keys:
            conv_active_weights.setdefault(key, list())
        key += 1
    
    if use_predictions_from_frz_predictor_to_frz_layers:
      if use_post_processing_window_for_frz_predictor:
        # Begin making decisions in here
        for p_index, p in enumerate(conv_active):
            if len(frz_decisions_per_layer[p]) < post_processing_window_size:
                print(f"Waiting on predictions for Conv# {p}, at length {len(frz_decisions_per_layer[p])}")
            else:
                prev_decisions = frz_decisions_per_layer[p][-post_processing_window_size:]
                num_frz_predictions = sum([1 if prev_pred == 1 else 0 for prev_pred in prev_decisions])
                
                if num_frz_predictions >= threshold_for_post_processing_frz_decisions_in_window:
                    conv_freeze_list.append(p)
                    print(f"Post-Processing: Layer {track_conv_frozen[p][0]}, Frz Epoch {epoch}, Conv# {p}")
                else:
                    if use_linear_restriction_for_layer_freezing:
                        print(f"Post-Processing: Stopped processing at Conv# {p}")
                        break
      
      bn_freeze_list = conv_freeze_list.copy()

      for i2 in conv_freeze_list:
        conv_frozen.append(i2)  # Record the frozen layer
        conv_active.remove(i2)  # Remove the corresponding entry from the list and dictionary
        conv_active_weights.pop(i2)
        for params in conv_layer[i2].parameters():
          params.requires_grad = False
          conv_layer_param.setdefault(i2, list())
          conv_layer_param[i2].append(params.data.clone().detach())
      
      # Check to see that the network architecture does have BN layers
      if len(bn_layer) != 0:
        for i2 in bn_freeze_list:
          bn_frozen.append(i2)
          for params in bn_layer[i2].parameters():
            params.requires_grad = False
            bn_layer_param.setdefault(i2, list())
            bn_layer_param[i2].append(params.data.clone().detach())

    if accuracy > best_acc:
      checkpoint = net.state_dict()
      torch.save(checkpoint, f"{name_of_experiment}/seed_{seed}/best_model.pt")
      print("New Best Testing Accuracy:", accuracy)
      best_acc = accuracy

  with open(f"{name_of_experiment}/seed_{seed}/fully_trained_reference_model_vgg11_metrics.pkl", "wb") as f:
    pickle.dump((accuracy_list, training_loss_list, testing_loss_list), f)

  with open(f"{name_of_experiment}/seed_{seed}/vgg11_cka_values_across_epochs.pkl", "wb") as f:
    pickle.dump(tracker_of_cka_values_across_epochs, f)
  with open(f"{name_of_experiment}/seed_{seed}/cka_window_values_across_epochs.pkl", "wb") as f:
    pickle.dump(tracker_of_cka_window_values_across_epochs, f)
  with open(f"{name_of_experiment}/seed_{seed}/cka_freeze_layer_decisions.pkl", "wb") as f:
    pickle.dump(cka_freeze_layer_decisions, f)
  with open(f"{name_of_experiment}/seed_{seed}/vgg11_cka_freeze_configuration.pkl", "wb") as f:
    pickle.dump(cka_freeze_layer_configuration, f)
  # Save the output of freezing configuration
  with open(f"{name_of_experiment}/seed_{seed}/frz_predictor_track_conv_frozen.pkl", "wb") as f:
    pickle.dump(track_conv_frozen, f)

if __name__ == "__main__":
    args = parse_args()
    configuration_file = args.configuration_file

    try:
        with open(configuration_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Configuration file not found.")
    except json.JSONDecodeError as e:
        print(f"Invalid Configuration for JSON: {e}")

    config["device"] = args.device
    config["seed"] = args.seed

    main(config)

