import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.autoaugment import AutoAugment, RandAugment, AutoAugmentPolicy
import pickle
import numpy as np
import random
import gc
import math
from torchvision.transforms import v2
from CKA import CKA
from utils import random_sample, is_cnn_layer, is_bn_layer, soft_cross_entropy, WarmUpLR
from MambaFRZ import initialize_mamba2_predictor
from SmartFRZ import initialize_smartfrz_predictor
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # needed for full determinism in some CUDA ops
    
def main(args):
  best_acc = 0.0

  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # Create generator for DataLoader
  g = torch.Generator()
  g.manual_seed(args.seed)

  os.makedirs(f"{args.name_of_experiment}", exist_ok=True)
  # The location of the metrics for the experiment
  os.makedirs(f"{args.name_of_experiment}/seed_{args.seed}", exist_ok=True)

    
  device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    
  ## -----------------------------------------
  ## MAMBAFRZ Code Start!
  ## -----------------------------------------
  window_size = args.window_size
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

  # Define predictor
  feature_dim = args.re_size
  mlp_hid_channel = 256 # Hidden Size for MLP
  mlp_out_channel = 2 # Output Size, confidence for either freezing or not for MLP
  ssm_state_expansion_factor = 16 # keep it small for now
  predictor = initialize_mamba2_predictor(feature_dim, ssm_state_expansion_factor, mlp_hid_channel, mlp_out_channel)
  predictor_path = args.frz_predictor_path
  predictor.load_state_dict(torch.load(predictor_path, map_location=device))
  predictor = predictor.to(device)
  ## -----------------------------------------
  ## MAMBAFRZ Code End!
  ## -----------------------------------------


  transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    AutoAugment(AutoAugmentPolicy.CIFAR10),  # Apply AutoAugment
    RandAugment(),  # Randomly chosen augmentations
    transforms.ToTensor(),
    transforms.Normalize((0.5070758, 0.4865503, 0.44091913), (0.26733428, 0.25643846, 0.27615047)),
  ])

  transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070758, 0.4865503, 0.44091913), (0.26733428, 0.25643846, 0.27615047)),
  ])

  trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transforms_train)
  testset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=transforms_test)

  num_classes = 100
  model = resnet50(weights=None)
  # Modify the first convolution layer to accept 3x32x32 inputs
  model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  # Adjust the maxpool layer (since CIFAR-100 is small)
  model.maxpool = nn.Identity()  # Remove max pooling
  # Modify the last fully connected layer for CIFAR-100
  num_features = model.fc.in_features
  model.fc = nn.Linear(num_features, num_classes)

  net = model
  net.to(device)

    
  ## -----------------------------------------
  ## MAMBAFRZ Code Start!
  ## -----------------------------------------
  for name, layer in net.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
      conv_layer.append(layer)
    elif isinstance(layer, torch.nn.BatchNorm2d):
      bn_layer.append(layer)
  # Prepare key data structure needed for calculating CKA
  key = 0
  # Track which convolutional layer has been frozen to measure TFLOPs
  track_conv_frozen = {}
  for name, layer in net.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
      key_list.append(key)
      conv_active_weights.setdefault(key, list())
      track_conv_frozen.setdefault(key, [name, []])
      conv_active.append(key)
      bn_active.append(key)
      key += 1
  ## -----------------------------------------
  ## MAMBAFRZ Code End!
  ## -----------------------------------------

  fully_trained_model = resnet50(weights=None)
  fully_trained_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  fully_trained_model.maxpool = nn.Identity()
  num_features = fully_trained_model.fc.in_features
  fully_trained_model.fc = nn.Linear(num_features, num_classes)
  fully_trained_model = fully_trained_model.to(device)
  torch_state = torch.load(args.fully_trained_reference_model, map_location=device)
  fully_trained_model.load_state_dict(torch_state)
  fully_trained_reference_model = fully_trained_model

  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

  def seed_worker(worker_id):
    """
    Ensure reproducibility for each worker in DataLoader.
    """
    worker_seed = args.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

  num_workers = min(8, os.cpu_count() // 8)  # Use half of available cores
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
  testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
  cka_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=num_workers, drop_last=True)

  optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
  train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs - args.warm, eta_min=1e-3) # learning rate decay
  iter_per_epoch = len(trainloader)
  warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

  tracker_of_layers_randomly_sampled_input_weights = {}
  tracker_of_cka_values_across_epochs = {}
  cka_freeze_layer_decisions = {}
  cka_freeze_layer_configuration = {}
  tracker_of_cka_window_values_across_epochs = {}
    
  set_of_all_layer_names = set()
  counter_for_cnn_layers = args.number_of_cnn_layers
    
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
  

  accuracy_list = []
  training_loss_list = []
  testing_loss_list = []

  cutmix = v2.CutMix(num_classes=num_classes)
  mixup = v2.MixUp(num_classes=num_classes)
  cutmix_or_mixup = transforms.RandomChoice([cutmix, mixup])

  for epoch in range(args.epochs):
    if epoch >= args.warm:
      train_scheduler.step()

    net.train()
    train_running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      ## -----------------------------------------
      ## MAMBAFRZ Code Start!
      ## -----------------------------------------
      if len(conv_active):
        num_training_samples = len(trainloader.dataset)
        if i % int((num_training_samples/args.batch_size) / args.window_size) == 0:
          # Get the weights of current models, only for active layers
          for layer_index in conv_active:
            sampled_weights = random_sample(conv_layer[layer_index].weight.clone().detach().cpu().reshape(-1).unsqueeze(0), args.re_size)
            conv_active_weights[layer_index].append(sampled_weights)
            
            layer_name = track_conv_frozen[layer_index][0]
            if layer_name in set_of_all_layer_names:
                tracker_of_layers_randomly_sampled_input_weights[layer_name].append(sampled_weights.cpu())

      ## -----------------------------------------
      ## MAMBAFRZ Code End!
      ## -----------------------------------------
      inputs, labels = data[0].to(device), data[1].to(device)
      inputs, labels = cutmix_or_mixup(inputs, labels)

      optimizer.zero_grad()

      outputs = net(inputs)
      loss = soft_cross_entropy(outputs, labels)
      loss.backward()
      optimizer.step()

      train_running_loss += loss.item()

      if epoch < args.warm:
        warmup_scheduler.step()

    train_running_loss /= len(trainloader)
    
    file_path = f"{args.name_of_experiment}/smartfrz_input_data/seed_{args.seed}_{args.cuda_device}/epoch_{epoch}.pkl"
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
      print(f"Comparing for {model_layer} at epoch {epoch}")
      cka = CKA(net, fully_trained_reference_model, model1_name=f'ResNet50_{epoch}_Epoch', model2_name='ResNet50_Fully_Trained', model1_layers=[model_layer], model2_layers=[model_layer], device=device)
      with torch.no_grad():
        cka.compare(cka_loader, None, num_times_iterate_over_test_dataset=1)
    
      output_cka_dict = cka.export()
      tracker_of_cka_values_across_epochs[model_layer].append(output_cka_dict['CKA'][0].item())
      tracker_of_cka_window_values_across_epochs[model_layer].append(output_cka_dict['CKA'][0].item())

      if len(tracker_of_cka_window_values_across_epochs[model_layer]) > args.moving_window:
        # Remove previous CKA value if we have over the window size
        tracker_of_cka_window_values_across_epochs[model_layer] = tracker_of_cka_window_values_across_epochs[model_layer][args.stride:]

      if len(tracker_of_cka_window_values_across_epochs[model_layer]) == args.moving_window:
        # Begin checking variance threshold if we reached that point, use sample variance for unbiased estimate
        window_variance = np.var(tracker_of_cka_window_values_across_epochs[model_layer], ddof=1)

        current_cka_value = output_cka_dict['CKA'][0].item()
        if (window_variance < args.variance_threshold) and current_cka_value > args.cka_value_cutoff:
          cka_freeze_layer_decisions[model_layer].append(1)
          if is_previous_layer_frozen:
            cka_freeze_layer_configuration[model_layer] = epoch
            print(f"Froze layer {model_layer} at epoch {epoch}")
          
          if is_previous_layer_frozen and args.similarity_guided_training:  
            # Signal to help freeze the corresponding batch normalization layer
            froze_cnn = False
            for name, module in net.named_modules():
              if is_cnn_layer(module) and name == model_layer:
                froze_cnn = True
                module.weight.requires_grad = False
                if hasattr(module, 'bias') and module.bias is not None:
                  module.bias.requires_grad = False
              elif is_bn_layer(module) and froze_cnn:
                module.weight.requires_grad = False
                if hasattr(module, 'bias') and module.bias is not None:
                  module.bias.requires_grad = False
                # Break out since we have already frozen corresponding batch normalization layer
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
      for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)

        totals += labels.size(0)
        correct += (predicted == labels).sum().item()
        testing_loss += criterion(outputs, labels).item()

    testing_loss /= len(testloader)
    accuracy = correct / totals
    print("Epoch: ", epoch, "Accuracy: ", accuracy, "Training Loss: ", train_running_loss, "Testing Loss: ", testing_loss)

    # Save metrics
    testing_loss_list.append(testing_loss)
    accuracy_list.append(accuracy)
    training_loss_list.append(train_running_loss)
                                                    
                                                    
    ## -----------------------------------------
    ## MAMBAFRZ Code Start!
    ## -----------------------------------------                                                
    # At the end of each epoch, make freeze predictions and record them
    # not going to freeze right now, but later on I will
    conv_freeze_list = []
    for p_index, p in enumerate(conv_active):
      freeze_input = conv_active_weights[p][0]
      for index, weights in enumerate(conv_active_weights[p]):
        if index == 0:
          continue
        freeze_input = torch.cat((freeze_input, weights), 0)
        if index >= args.window_size - 1:
          break
      freeze_input = freeze_input.unsqueeze(0)
      freeze_input = freeze_input.to(device)
      # Predict the freezing decision
      pred = predictor(freeze_input)
      prediction_for_freezing = torch.argmax(pred).item()
      if prediction_for_freezing == 1:
        conv_freeze_list.append(p)
        track_conv_frozen[p][1].append(1)
        print(f"Layer {track_conv_frozen[p][0]} frz predictor at epoch {epoch}, conv # {p}")
      else:
        track_conv_frozen[p][1].append(0)
        if args.use_linear_restriction:
            # Make all subsequent p indices append a 0
            for extended_p_index, extended_p in enumerate(conv_active):
                if extended_p_index <= p_index:
                    continue
                else:
                    track_conv_frozen[extended_p][1].append(0)
            break
    
    # After each epoch, we need to delete the weights contained previously
    del conv_active_weights
    conv_active_weights = dict()
    key = 0
    for name, layer in net.named_modules():
      if isinstance(layer, torch.nn.Conv2d):
        conv_active_weights.setdefault(key, list())
        key += 1
    
    if args.frz_from_frz_predictor:
      bn_freeze_list = conv_freeze_list.copy()
      for i2 in conv_freeze_list:
        conv_frozen.append(i2)  # Record the frozen layer
        conv_active.remove(i2)  # Remove the corresponding entry from the list and dictionary
        conv_active_weights.pop(i2)
        for params in conv_layer[i2].parameters():
          params.requires_grad = False
          conv_layer_param.setdefault(i2, list())
          conv_layer_param[i2].append(params.data.clone().detach())
      
      for i2 in bn_freeze_list:
        bn_frozen.append(i2)  # Record the frozen layer
        for params in bn_layer[i2].parameters():
          params.requires_grad = False
          bn_layer_param.setdefault(i2, list())
          bn_layer_param[i2].append(params.data.clone().detach())              
    ## -----------------------------------------
    ## MAMBAFRZ Code End!
    ## ----------------------------------------- 

    if accuracy > best_acc:
      checkpoint = net.state_dict()
      torch.save(checkpoint, f"{args.name_of_experiment}/seed_{args.seed}/best_model.pt")
      print("New best accuracy:", accuracy)
      best_acc = accuracy

  with open(f"{args.name_of_experiment}/seed_{args.seed}/fully_trained_reference_model_resnet50_metrics.pkl", "wb") as f:
    pickle.dump((accuracy_list, training_loss_list, testing_loss_list), f)

  with open(f"{args.name_of_experiment}/seed_{args.seed}/resnet50_cka_values_across_epochs.pkl", "wb") as f:
    pickle.dump(tracker_of_cka_values_across_epochs, f)
  with open(f"{args.name_of_experiment}/seed_{args.seed}/cka_window_values_across_epochs.pkl", "wb") as f:
    pickle.dump(tracker_of_cka_window_values_across_epochs, f)
  with open(f"{args.name_of_experiment}/seed_{args.seed}/cka_freeze_layer_decisions.pkl", "wb") as f:
    pickle.dump(cka_freeze_layer_decisions, f)
  with open(f"{args.name_of_experiment}/seed_{args.seed}/resnet50_cka_freeze_configuration.pkl", "wb") as f:
    pickle.dump(cka_freeze_layer_configuration, f)
  # Save the output of freezing configuration
  with open(f"{args.name_of_experiment}/seed_{args.seed}/frz_predictor_track_conv_frozen.pkl", "wb") as f:
    pickle.dump(track_conv_frozen, f)

class Arguments:
 def __init__(self, epochs=160, batch_size=128, seed=1, warm=10, fully_trained_reference_model="best_model.pt", re_size=1024, variance_threshold=0.0002, moving_window=20, cka_value_cutoff=0.3, stride=5, cuda_device="cuda:0", number_of_cnn_layers=1, name_of_experiment="experiment", similarity_guided_training=False, window_size=30, frz_predictor_path="frz_predictor.pt", frz_from_frz_predictor=False, use_linear_restriction=False):
  self.epochs = epochs
  self.batch_size = batch_size
  self.seed = seed
  self.warm = warm
  self.fully_trained_reference_model = fully_trained_reference_model
  self.re_size = re_size
  self.variance_threshold = variance_threshold
  self.moving_window = moving_window
  self.cka_value_cutoff = cka_value_cutoff
  self.stride = stride
  self.cuda_device = cuda_device
  self.number_of_cnn_layers = number_of_cnn_layers
  self.name_of_experiment = name_of_experiment
  self.similarity_guided_training = similarity_guided_training
  self.window_size = window_size
  self.frz_predictor_path = frz_predictor_path
  self.frz_from_frz_predictor = frz_from_frz_predictor
  self.use_linear_restriction = use_linear_restriction

################################################
##### Validation of New FRZ Predictor      #####
################################################
seed = 2088 # In-Distribution
context_window_size = 30
args = Arguments(moving_window=20, cka_value_cutoff=0.3, stride=5, variance_threshold=0.0002, cuda_device="cuda:0", name_of_experiment="mambafrz_20_conv_seed_25_experiment/validation_of_predictions_linear_restriction_plus_frz", fully_trained_reference_model="test_model_weights/best_model_test.pt", window_size=context_window_size, frz_predictor_path=f"mambafrz_20_conv_seed_25_experiment/training_data/context_window_{context_window_size}/checkpoints/mambafrz_trained_4.pth", seed=seed, number_of_cnn_layers=20, frz_from_frz_predictor=True, use_linear_restriction=True)
main(args)

################################################
###### Generate Data from Different Seeds ######
################################################
# num_seeds_to_generate = 5
# for i in range(num_seeds_to_generate):
#     rand_seed = random.randint(0, 10000) # or any seed range you prefer

#     # Actual running of the script afterwards
#     args = Arguments(moving_window=20, cka_value_cutoff=0.3, stride=5, variance_threshold=0.0002, cuda_device="cuda:0", name_of_experiment="mambafrz_20_conv_seed_25_experiment", fully_trained_reference_model="test_model_weights/best_model_test.pt", window_size=30, frz_predictor_path="test_model_weights/test_mamba2_context_window_30.pth", seed=rand_seed, number_of_cnn_layers=20)
#     print(f"Running iteration {i+1} with seed {rand_seed}")
#     main(args)