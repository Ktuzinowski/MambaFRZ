import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy, RandAugment

def get_dataloaders_for_dataset(dataset, batch_size, cka_batch_size, num_workers, worker_init_fn=None, generator=None):
    if dataset == "cifar100":
        return get_dataloaders_for_cifar100(batch_size, cka_batch_size, num_workers, worker_init_fn, generator)
    elif dataset == "cifar10":
        return get_dataloaders_for_cifar10(batch_size, cka_batch_size, num_workers, worker_init_fn, generator)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def get_dataloaders_for_cifar100(batch_size, cka_batch_size, num_workers, worker_init_fn, generator):
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
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, worker_init_fn=worker_init_fn, generator=generator)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, generator=generator)
    cka_loader = torch.utils.data.DataLoader(testset, batch_size=cka_batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    return train_loader, test_loader, cka_loader

def get_dataloaders_for_cifar10(batch_size, cka_batch_size, num_workers, worker_init_fn, generator):
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(AutoAugmentPolicy.CIFAR10),  # Apply AutoAugment
        RandAugment(),  # Randomly chosen augmentations
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transforms_train)
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transforms_test)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, worker_init_fn=worker_init_fn, generator=generator)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init_fn, generator=generator)
    cka_loader = torch.utils.data.DataLoader(testset, batch_size=cka_batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    return train_loader, test_loader, cka_loader