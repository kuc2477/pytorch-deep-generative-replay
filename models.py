from functools import reduce
import torch
from torch import nn, autograd
from torch.autograd import Variable
import gan
import dgr
import utils


class WGAN(dgr.Generator):
    def __init__(self, z_size,
                 image_size, image_channel_size,
                 c_channel_size, g_channel_size):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.c_channel_size = c_channel_size
        self.g_channel_size = g_channel_size

        # components
        self.critic = gan.Critic(
            image_size=self.image_size,
            image_channel_size=self.image_channel_size,
            channel_size=self.c_channel_size,
        )
        self.generator = gan.Generator(
            z_size=self.z_size,
            image_size=self.image_size,
            image_channel_size=self.image_channel_size,
            channel_size=self.g_channel_size,
        )

        # training related components that should be set before training.
        self.generator_optimizer = None
        self.critic_optimizer = None
        self.critic_updates_per_generator_update = None
        self.lamda = None

    def train_a_batch(self, x, y):
        # run the critic and backpropagate the errors.
        for _ in range(self.critic_updates_per_generator_update):
            self.critic_optimizer.zero_grad()
            z = self._noise(x.size(0))
            c_loss, g = self._c_loss(x, z, return_g=True)
            c_loss_gp = c_loss + self._gradient_penalty(x, g, self.lamda)
            c_loss_gp.backward()
            self.critic_optimizer.step()

        # run the generator and backpropagate the errors.
        self.generator_optimizer.zero_grad()
        z = self._noise(x.size(0))
        g_loss = self._g_loss(z)
        g_loss.backward()
        self.generator_optimizer.step()

        return {'c_loss': c_loss, 'g_loss': g_loss}

    def sample(self, size):
        return self.generator(self._noise(size))

    def set_generator_optimizer(self, optimizer):
        self.generator_optimizer = optimizer

    def set_critic_optimizer(self, optimizer):
        self.critic_optimizer = optimizer

    def set_critic_updates_per_generator_update(self, k):
        self.critic_updates_per_generator_update = k

    def set_lambda(self, l):
        self.lamda = l

    def _noise(self, size):
        z = Variable(torch.randn(size, self.z_size)) * .1
        return z.cuda() if self._is_on_cuda() else z

    def _c_loss(self, x, z, return_g=False):
        g = self.generator(z)
        c_x = self.critic(x).mean()
        c_g = self.critic(g).mean()
        l = -(c_x-c_g)
        return (l, g) if return_g else l

    def _g_loss(self, z, return_g=False):
        g = self.generator(z)
        l = -self.critic(g).mean()
        return (l, g) if return_g else l

    def _gradient_penalty(self, x, g, lamda):
        assert x.size() == g.size()
        a = torch.rand(x.size(0), 1)
        a = a.cuda() if self._is_on_cuda() else a
        a = a\
            .expand(x.size(0), x.nelement()//x.size(0))\
            .contiguous()\
            .view(
                x.size(0),
                self.image_channel_size,
                self.image_size,
                self.image_size
            )
        interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True)
        c = self.critic(interpolated)
        gradients = autograd.grad(
            c, interpolated, grad_outputs=(
                torch.ones(c.size()).cuda() if self._is_on_cuda() else
                torch.ones(c.size())
            ),
            create_graph=True,
            retain_graph=True,
        )[0]
        return lamda * ((1-gradients.norm(2, dim=1))**2).mean()


class CNN(dgr.Solver):
    def __init__(self,
                 image_size,
                 image_channel_size, classes,
                 depth, channel_size, reducing_layers=3):
        # configurations
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.classes = classes
        self.depth = depth
        self.channel_size = channel_size
        self.reducing_layers = reducing_layers

        # layers
        self.layers = nn.ModuleList([nn.Conv2d(
            self.image_channel_size, self.channel_size//(2**(depth-2)),
            3, 1, 1
        )])

        for i in range(self.depth-2):
            previous_conv = [
                l for l in self.layers if
                isinstance(l, nn.Conv2d)
            ][-1]
            self.layers.append(nn.Conv2d(
                previous_conv.out_channels,
                previous_conv.out_channels * 2,
                3, 1 if i >= reducing_layers else 2, 1
            ))
            self.layers.append(nn.BatchNorm2d())
            self.layers.append(nn.ReLU())

        self.layers.append(utils.LambdaModule(lambda x: x.view(x.size(0), -1)))
        self.layers.append(nn.Linear(self.channel_size, self.classes))

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)
