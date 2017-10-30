from torchvision import datasets, transforms
from torchvision.transforms import ImageOps


def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image


def get_dataset(name, train=True, permutation=None):
    dataset = TRAIN_DATASETS[name] if train else TEST_DATASETS[name]
    dataset.transform = transforms.Compose([
        dataset.transform,
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])
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
    transforms.Lambda(lambda x: ImageOps.colorize(x, (0, 0, 0), (1, 1, 1))),
    transforms.Pad(2),
    transforms.ToTensor(),
]

_CIFAR_TRAIN_TRANSFORMS = _CIFAR_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]

_SVHN_IMAGE_SIZE = 32
_SVHN_TRAIN_TRANSFORMS = _SVHN_TEST_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Scale(_SVHN_IMAGE_SIZE),
    transforms.CenterCrop(_SVHN_IMAGE_SIZE)
]


TRAIN_DATASETS = {
    'mnist': datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
    ),
    'mnist-color': datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose(_MNIST_COLORIZED_TRAIN_TRANSFORMS)
    ),
    'cifar10': datasets.CIFAR10(
        './datasets/cifar10', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'cifar100': datasets.CIFAR100(
        './datasets/cifar100', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    ),
    'svhn': datasets.SVHN(
        './datasets/svhn', split='train', download=True,
        transform=transforms.Compose(_SVHN_TRAIN_TRANSFORMS)
    ),
}


TEST_DATASETS = {
    'mnist': datasets.MNIST(
        './datasets/mnist', train=False,
        transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
    ),
    'mnist-color': datasets.MNIST(
        './datasets/mnist', train=False, download=True,
        transform=transforms.Compose(_MNIST_COLORIZED_TEST_TRANSFORMS)
    ),
    'cifar10': datasets.CIFAR10(
        './datasets/cifar10', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'cifar100': datasets.CIFAR100(
        './datasets/cifar100', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    'svhn': datasets.SVHN(
        './datasets/svhn', split='test', download=True,
        transform=transforms.Compose(_SVHN_TEST_TRANSFORMS)
    ),
}


DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist-color': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'svhn': {'size': _SVHN_IMAGE_SIZE, 'channels': 3, 'classes': 10}
}
