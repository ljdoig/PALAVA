import os
import torch 
from .palava import Pathway
from .utils import save_tensor

STDNRML = torch.distributions.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))

def norm(mean=0, sd=1):
  return mean + sd * STDNRML.sample()
  
class DataSampler(object):
  def __init__(
    self, num_facs, num_attrs, num_obs, overlap=False, dense_fac=False
  ):
    self.num_facs = num_facs
    self.num_attrs = num_attrs
    self.num_obs = num_obs
    self.overlap=overlap
    self.dense_fac=dense_fac
    self.create_data_structure()

  def create_data_structure(self):
    # Create loadings structure matrix
    num_sparse_facs = self.num_facs
    if self.dense_fac:
      num_sparse_facs -= 1
    chunk_size = self.num_attrs / num_sparse_facs

    self.loadings_struct = torch.ones(self.num_attrs, self.num_facs)
    for fac in range(num_sparse_facs):
      for gene in range(0, int(chunk_size * fac)):
        self.loadings_struct[[gene], [fac]] = torch.zeros(1)
      for gene in range(int(chunk_size * (fac + 1)), self.num_attrs):
        self.loadings_struct[[gene], [fac]] = torch.zeros(1)

    # Create activations structure matrix
    if self.overlap:
      non_included_size = (
        self.num_obs * (num_sparse_facs - 2) // (num_sparse_facs - 1)
      )
    else:
      non_included_size = (
        self.num_obs * (num_sparse_facs - 1) // num_sparse_facs
      )
    gap_size = non_included_size // (num_sparse_facs - 1)

    self.activations_struct = torch.ones(self.num_facs, self.num_obs)
    for fac in range(num_sparse_facs):
      for obs in range(0, gap_size * fac):
        self.activations_struct[[fac], [obs]] = torch.zeros(1)
      for obs_offset in range(0, gap_size * (num_sparse_facs - fac - 1)):
        col = self.num_obs - obs_offset - 1
        self.activations_struct[[fac], [col]] = torch.zeros(1)

    # Expand matrices to get approx data shape
    self.data_struct = self.loadings_struct @ self.activations_struct

  def sample_from_data_structure(
    self, zero_sd, nonzero_sd, dense_fac_sd, noise
  ):
    # Duplicate underlying structure matrices before sampling
    self.loadings = self.loadings_struct.clone()
    self.activations = self.activations_struct.clone()

    # Sample
    for mat in [self.loadings, self.activations]:
      for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
          if mat[row, col] == 1:
            # dense pathway is always last row or column, and has smaller sd
            if (self.dense_fac and (
              (col == mat.shape[1] - 1 and mat.shape[1] == self.num_facs) or 
              (row == mat.shape[0] - 1 and mat.shape[0] == self.num_facs))):
              mat[[row], [col]] = norm(sd=dense_fac_sd)
            else:
              mat[[row], [col]] = norm(sd=nonzero_sd)
          else:
            mat[[row], [col]] = norm(sd=zero_sd)
    self.data = self.loadings @ self.activations

    for row in range(self.data.shape[0]):
      for col in range(self.data.shape[1]):
        self.data[[row], [col]] += norm(sd=noise)
  
  def get_samples_by_row(self):
    return self.data.t()

  # get pathway information from loadings matrix - can requrest error 
  def get_pathways(
    self, num_paths, add_error, false_pos=0, false_neg=0, error_in_block=False
  ):
    assert num_paths == len(add_error)
    pathways = []
    for path_index, error in enumerate(add_error):
      loading = self.loadings_struct[:,path_index].clone()
      if error:
        if error_in_block:
          num_true = sum(loading)
          num_false = len(loading) - num_true
          num_false_neg = round(float(false_neg * num_true))
          num_false_pos = round(float(false_pos * num_false))
          for i in range(len(loading)):
            if loading[i] and num_false_neg > 0:
              loading[i] = 0
              num_false_neg -= 1
            elif (not loading[i]) and num_false_pos > 0:
              loading[i] = 1
              num_false_pos -= 1
        else:
          for i in range(len(loading)):
            if loading[i]:
              loading[i] = 1 - false_neg
            else:
              loading[i] = false_pos
          loading = torch.bernoulli(torch.Tensor(loading))
      pathways.append(Pathway(loading)) 
    return pathways

  # Visualise and save underlying structure and samples
  def save_matrices(self, dir, structure=False, samples=False, overwrite=False):
    os.makedirs(dir, exist_ok=overwrite)

    if structure:
      save_tensor(
        self.loadings_struct, "pathway", "gene", dir, "loadings_struct",
        title="loadings structure"
      )
      save_tensor(
        self.activations_struct, "cell", "pathway", dir,
        "activations_struct", title="activations structure"
      )
      save_tensor(
        self.data_struct, "cell", "gene", dir, "data_struct",
         title="overall structure"
      )
    if samples:
      save_tensor(
        self.loadings, "pathway", "gene", dir, "loadings", 
        title="sampled loadings", cmap="bwr", ind_norm=True
      )
      save_tensor(
        self.activations, "cell", "pathway", dir, "activations", 
        title="sampled activations", cmap="bwr", ind_norm=True
      )
      save_tensor(
        self.data, "cell", "gene", dir, "data", title="data",
        cmap="bwr", ind_norm=True
      )