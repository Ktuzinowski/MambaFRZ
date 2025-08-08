import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg11, vgg16, vgg11_bn, resnet50

def get_model_for_dataset(model_name, dataset_name):
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        return get_models_for_cifar(model_name, dataset_name)
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

def get_models_for_cifar(model_name, dataset_name):
    if dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    
    if model_name == "vgg11":
        model = vgg11(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif model_name == "vgg16":
        model = vgg16(weights=None)
        model.features[4] = nn.Identity()
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif model_name == "vgg11_bn":
        model = vgg11_bn(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        classifier = nn.Sequential(
          nn.Linear(512, 256),
          nn.ReLU(inplace=True),
          nn.Dropout(0.5),
          nn.Linear(256, num_classes)
        )
        model.classifier = classifier
    elif model_name == "resnet50":
        model = resnet50(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    return model