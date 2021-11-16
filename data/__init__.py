from .poisoned_dataset import PoisonedDataset

from torch.utils.data import DataLoader, random_split
from torchvision import datasets,transforms


def load_init_data(dataname, device, download, dataset_path):
    if dataname == 'mnist':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transforms = transforms.Compose([
            # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            transforms.RandomCrop(size=(32,32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    
        train_data = datasets.CIFAR10(root=dataset_path, train=True, transform=train_transforms, download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, transform=test_transforms, download=download)
    return train_data, test_data


def create_backdoor_data_loader(dataname, train_data, test_data, trigger_label, posioned_portion, batch_size, device):
    train_data    = PoisonedDataset(train_data, trigger_label, portion=posioned_portion, mode="train", device=device, dataname=dataname)
    test_data_ori = PoisonedDataset(test_data,  trigger_label, portion=0,                mode="test",  device=device, dataname=dataname)
    test_data_tri = PoisonedDataset(test_data,  trigger_label, portion=1,                mode="test",  device=device, dataname=dataname)

    train_data_loader       = DataLoader(dataset=train_data,    batch_size=batch_size, shuffle=True)
    test_data_ori_loader    = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True)
    test_data_tri_loader    = DataLoader(dataset=test_data_tri, batch_size=batch_size, shuffle=True) # shuffle 随机化

    return train_data_loader, test_data_ori_loader, test_data_tri_loader
