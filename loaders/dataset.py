import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from constants import INPUT_PATH, TEST_PATH, SIZE, BATCH_SIZE
from loaders.car_detection_dataset import CarDataset
from loaders.transf import Rescale, ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_classification_dataset(input=INPUT_PATH, test=TEST_PATH, size=SIZE, bsize=BATCH_SIZE):
    print("Loading training dataset from ", input)
    print("Loading testing dataset from ", test)
    print("Image size: {}; Batch size: {}".format(size, bsize))
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.ImageFolder(input, transform=transform)
    testset = datasets.ImageFolder(test, transform=transform)

    train_loader, validation_loader, test_loader = get_loaders(trainset, testset, bsize)

    return train_loader, validation_loader, test_loader


def get_car_detection_dataset(size=SIZE, bsize=BATCH_SIZE):
    trainset = CarDataset(
        csv_file='../stanford-car-dataset-by-classes-folder/anno_train.csv',
        root_dir='../stanford-car-dataset-by-classes-folder/car_data/all_train',
        transform=transforms.Compose([
            Rescale(size),
            ToTensor()
        ]))

    testset = CarDataset(
        csv_file='../stanford-car-dataset-by-classes-folder/anno_test.csv',
        root_dir='../stanford-car-dataset-by-classes-folder/car_data/all_test',
        transform=transforms.Compose([
            Rescale(size),
            ToTensor()
        ]))

    train_loader, validation_loader, test_loader = get_loaders(trainset, testset, bsize)

    return train_loader, validation_loader, test_loader


def create_validation_sampler(dataset, validation_split=0.1):
    dataset_len = len(dataset)
    indices = list(range(dataset_len))
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    return train_sampler, validation_sampler


def get_loaders(trainset, testset, bsize):
    train_loader = DataLoader(trainset, batch_size=bsize,
                              shuffle=True, num_workers=4)

    test_sampler, validation_sampler = create_validation_sampler(testset, validation_split=0.1)

    test_loader = DataLoader(testset, batch_size=bsize, sampler=test_sampler,
                             shuffle=False, num_workers=4)
    validation_loader = DataLoader(testset, batch_size=bsize, sampler=validation_sampler,
                                   shuffle=False, num_workers=4)

    return train_loader, validation_loader, test_loader
