import math
import torch
import itertools
from torch.autograd import Variable

LOG2PI = torch.log(torch.FloatTensor([2 * math.pi]))[0]

class Normal(object):
  def __init__(self, mu, sigma):
    assert mu.size() == sigma.size()
    self.mu = mu
    self.sigma = sigma

  def size(self, *args, **kwargs):
    return self.mu.size(*args, **kwargs)

  def sample(self):
    eps = torch.randn(self.mu.size()).type_as(self.mu.data)
    return self.mu + self.sigma * Variable(eps)

  def logprob(self, x):
    return torch.sum(
      -0.5 * LOG2PI
      - torch.log(self.sigma)
      -0.5 * torch.pow((x - self.mu) / self.sigma, 2)
    )

  def expand(self, size):
    return Normal(
      self.mu.expand(size),
      self.sigma.expand(size)
    )

class NormalNet(object):
  def __init__(self, mu_net, sigma_net):
    self.mu_net = mu_net
    self.sigma_net = sigma_net

  def __call__(self, x):
    return Normal(self.mu_net(x), self.sigma_net(x))

  def parameters(self):
    return itertools.chain(
      self.mu_net.parameters(),
      self.sigma_net.parameters()
    )
  
  def cuda(self):
    self.mu_net.cuda()
    self.sigma_net.cuda()
