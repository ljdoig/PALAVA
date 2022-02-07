import torch
import os
import matplotlib
from matplotlib import pyplot as plt

def trace(x):
  print(x)
  return x

# See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
def KL_Normals(d0, d1):
  assert d0.mu.size() == d1.mu.size()
  sigma0_sqr = torch.pow(d0.sigma, 2)
  sigma1_sqr = torch.pow(d1.sigma, 2)
  return torch.sum(
    -0.5
    + (sigma0_sqr + torch.pow(d0.mu - d1.mu, 2)) / (2 * sigma1_sqr)
    + torch.log(d1.sigma)
    - torch.log(d0.sigma)
  )

class Lambda(torch.nn.Module):
  def __init__(self, func, extra_args=(), extra_kwargs={}):
    super(Lambda, self).__init__()
    self.func = func
    self.extra_args = extra_args
    self.extra_kwargs = extra_kwargs

  def forward(self, x):
    return self.func(x, *self.extra_args, **self.extra_kwargs)

# These two functions are to ensure plots are centred at 0
def get_combined_norm(tensor1, tensor2):
  # visualise the two tensors with the same scale
  min_val = min(torch.min(tensor1), torch.min(tensor2))
  max_val = max(torch.max(tensor1), torch.max(tensor2))
  # shift scale to have 0 in the middle
  return matplotlib.colors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)

def get_individual_norm(tensor):
  return matplotlib.colors.TwoSlopeNorm(
    vmin=torch.min(tensor), vcenter=0, vmax=torch.max(tensor)
  )

def save_tensor(
  tensor, xlab, ylab, dir, filename, scale_norm=None, ind_norm=False, title="",
  drop_x_ticks=False, drop_y_ticks=False, cmap = "viridis"
):
  fig, ax = plt.subplots()
  matplotlib.rcParams['image.interpolation'] = 'nearest'
  if len(tensor.shape) == 1:
    tensor = torch.atleast_2d(tensor.contiguous())
  
  if ind_norm:
    scale_norm = get_individual_norm(tensor)

  if not scale_norm is None:
    ax.imshow(tensor, norm=scale_norm, cmap=cmap)
  else:
    ax.imshow(tensor, cmap=cmap)

  # set aspect ratio to 1, unless matrix is very thin then stretch by 4
  im = ax.get_images()
  extent =  im[0].get_extent()
  if tensor.shape[0] in [1, 2]:
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/4)
  elif tensor.shape[1] == [1, 2]:
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/(1/4))
  else:
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2])))
    
  ax.set_xlabel(xlab)
  ax.set_ylabel(ylab)
  ax.set_title(title)

  if drop_x_ticks:
    ax.set_xticks([])
  if drop_y_ticks:
    ax.set_yticks([])

  os.makedirs(dir, exist_ok=True)
  plt.savefig(f'{dir}/{filename}.png', format='png', dpi=600)
