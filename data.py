import copy
import math
from torchvision import datasets, transforms
from torchvision.transforms import ImageOps
from torch.utils.data import ConcatDataset


def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    return image.view(c, h, w)


def _colorize_grayscale_image(image):
    return ImageOps.colorize(image, (0, 0, 0), (255, 255, 255))


def get_dataset(name, train=True, permutation=None, capacity=None):
    dataset = (TRAIN_DATASETS[name] if train else TEST_DATASETS[name])()
    dataset.transform = transforms.Compose([
        dataset.transform,
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])

    if capacity is not None and len(dataset) < capacity:
        return ConcatDataset([
            copy.deepcopy(dataset) for _ in
            range(math.ceil(capacity / len(dataset)))
        ])
    else:
        return dataset


_MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_MNIST_COLORIZED_TRAIN_TRANSFORMS = _MNIST_COLORIZED_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_CIFAR_TRAIN_TRANSFORMS = _CIFAR_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]

_SVHN_TRAIN_TRANSFORMS = _SVHN_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]
_SVHN_TARGET_TRANSFORMS = [
    transforms.Lambda(lambda y: y % 10)
]


TRAIN_DATASETS = {
    'mnist': lambda: datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'mnist-color': lambda: datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_COLORIZED_TRAIN_TRANSFORMS)
    ),
    'cifar10': lambda: datasets.CIFAR10(
        './datasets/cifar10', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'cifar100': lambda: datasets.CIFAR100(
        './datasets/cifar100', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'svhn': lambda: datasets.SVHN(
        './datasets/svhn', split='train', download=True,
        transform=transforms.Compose(_SVHN_TRAIN_TRANSFORMS),
        target_transform=transforms.Compose(_SVHN_TARGET_TRANSFORMS),
    ),
}


TEST_DATASETS = {
    'mnist': lambda: datasets.MNIST(
        './datasets/mnist', train=False,
        transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
    ),
    'mnist-color': lambda: datasets.MNIST(
        './datasets/mnist', train=False, download=True,
        transform=transforms.Compose(_MNIST_COLORIZED_TEST_TRANSFORMS)
    ),
    'cifar10': lambda: datasets.CIFAR10(
        './datasets/cifar10', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'cifar100': lambda: datasets.CIFAR100(
        './datasets/cifar100', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'svhn': lambda: datasets.SVHN(
        './datasets/svhn', split='test', download=True,
        transform=transforms.Compose(_SVHN_TEST_TRANSFORMS),
        target_transform=transforms.Compose(_SVHN_TARGET_TRANSFORMS),
    ),
}


DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist-color': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},

}
