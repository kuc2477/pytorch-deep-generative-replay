import os
import os.path
import torchvision
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size, cuda=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save({'state': model.state_dict()}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=path
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])


def test_model(model, sample_size, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(
        model.sample_image(sample_size).data,
        path + '.jpg'
    )
    print('=> generated sample images at "{}".'.format(path))


def validate(model, dataset, test_size=256, cuda=False, verbose=True):
    data_loader = get_data_loader(dataset, 32, cuda=cuda)
    total_tested = 0
    total_correct = 0
    for data, labels in data_loader:
        # break on test size.
        if total_tested >= test_size:
            break
        # test the model.
        data = Variable(data).cuda() if cuda else Variable(data)
        labels = Variable(labels).cuda() if cuda else Variable(labels)
        scores = model(data)
        _, predicted = torch.max(scores, 1)
        # update statistics.
        total_correct += (predicted == labels).sum().data[0]
        total_tested += len(data)

    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


def xavier_initialize(model):
    modules = [m for n, m in model.named_modules() if 'conv' in n or 'fc' in n]
    parameters = [p for m in modules for p in m.parameters()]

    for p in parameters:
        if p.dim() >= 2:
            nn.init.xavier_normal(p)
        else:
            nn.init.constant(p, 0)


def gaussian_intiailize(model, std=.01):
    modules = [m for n, m in model.named_modules() if 'conv' in n or 'fc' in n]
    parameters = [p for m in modules for p in m.parameters()]

    for p in parameters:
        if p.dim() >= 2:
            nn.init.normal(p, std=std)
        else:
            nn.init.constant(p, 0)


class LambdaModule(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)
