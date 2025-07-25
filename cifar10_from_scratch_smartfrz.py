import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from SmartFRZ import initialize_smartfrz_predictor
import os
import numpy as np
from utils import random_sample, is_cnn_layer, is_bn_layer

if __name__ == "__main__":
    ## -----------------------------------------
    ## MAMBAFRZ Code Start!
    ## -----------------------------------------
    window_size = 30
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
    re_size = 1024
    in_channel = re_size
    hid_channel = 256
    out_channel = 64
    predictor = initialize_smartfrz_predictor(in_channel, hid_channel, out_channel)
    predictor_path = "mambafrz_20_conv_seed_25_experiment_same_seed_reference/training_data_0.0004_var_thresh/context_window_30/checkpoints/smartfrz_trained_0.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.load_state_dict(torch.load(predictor_path, map_location=device))
    predictor = predictor.to(device)

    use_linear_restriction = True
    frz_from_frz_predictor = True
    ## -----------------------------------------
    ## MAMBAFRZ Code End!
    ## -----------------------------------------

    # Configuration
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.1
    num_training_samples = 50000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = './resnet50_cifar10.pth'

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # CIFAR-10 Dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Define ResNet50 from scratch
    def get_resnet50_for_cifar10():
        model = resnet50(num_classes=10)
        # Modify the first convolution layer for CIFAR-10
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model

    model = get_resnet50_for_cifar10().to(device)

    ## -----------------------------------------
    ## MAMBAFRZ Code Start!
    ## -----------------------------------------
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_layer.append(layer)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            bn_layer.append(layer)
    # Prepare key data structure needed for calculating CKA
    key = 0
    # Track which convolutional layer has been frozen to measure TFLOPs
    track_conv_frozen = {}
    for name, layer in model.named_modules():
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

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    # Main training process
    best_acc = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for i , (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

             ## -----------------------------------------
             ## MAMBAFRZ Code Start!
             ## -----------------------------------------
            if len(conv_active):
                num_training_samples = len(trainloader.dataset)
                if i % int((num_training_samples/batch_size) / window_size) == 0:
                  # Get the weights of current models, only for active layers
                  for layer_index in conv_active:
                    sampled_weights = random_sample(conv_layer[layer_index].weight.clone().detach().cpu().reshape(-1).unsqueeze(0), re_size)
                    conv_active_weights[layer_index].append(sampled_weights)

                    layer_name = track_conv_frozen[layer_index][0]
        # At the end of each epoch, make freeze predictions and record them
        # not going to freeze right now, but later on I will
        conv_freeze_list = []
        for p_index, p in enumerate(conv_active):
          freeze_input = conv_active_weights[p][0]
          for index, weights in enumerate(conv_active_weights[p]):
            if index == 0:
              continue
            freeze_input = torch.cat((freeze_input, weights), 1)
            if index >= window_size - 1:
              break
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
            if use_linear_restriction:
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
        for name, layer in model.named_modules():
          if isinstance(layer, torch.nn.Conv2d):
            if key in current_keys:
                conv_active_weights.setdefault(key, list())
            key += 1

        if frz_from_frz_predictor:
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
        acc = 100. * correct / total
        print(f"Epoch {epoch} | Train Loss: {running_loss/total:.4f} | Acc: {acc:.2f}%")
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        print(f"Test Acc: {acc:.2f}%")
        scheduler.step()

        if acc > best_acc:
            print(f"Saving best model with acc: {acc:.2f}%")
            torch.save(model.state_dict(), save_path)
            best_acc = acc
