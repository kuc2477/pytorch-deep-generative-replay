from torchvision import datasets, transforms


_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]

_CIFAR_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
]

_CIFAR_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    ),
]


TRAIN_DATASETS = {
    'mnist': datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'cifar10': datasets.CIFAR10(
        './datasets/cifar10', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'cifar100': datasets.CIFAR100(
        './datasets/cifar100', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    )
}


TEST_DATASETS = {
    'mnist': datasets.MNIST(
        './datasets/mnist', train=False,
        transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
    ),
    'cifar10': datasets.CIFAR10(
        './datasets/cifar10', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'cifar100': datasets.CIFAR100(
        './datasets/cifar100', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    )
}


DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
}
