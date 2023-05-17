import math,pdb
import torch,pynvml
from torch.nn.functional import normalize
from time import time
import numpy as np
# device=torch.device("cuda:0")
def _kpp(data: torch.Tensor, k: int, sample_size: int = -1):
    """ Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    .. [2] scipy/cluster/vq.py: _kpp
    """
    batch_size=data.shape[0]
    if batch_size>sample_size:
        data = data[torch.randint(0, batch_size,[sample_size], device=data.device)]
    dims = data.shape[1] if len(data.shape) > 1 else 1
    init = torch.zeros((k, dims)).to(data.device)
    r = torch.distributions.uniform.Uniform(0, 1)
    for i in range(k):
        if i == 0:
            init[i, :] = data[torch.randint(data.shape[0], [1])]
        else:
            D2 = torch.cdist(init[:i, :][None, :], data[None, :], p=2)[0].amin(dim=0)
            probs = D2 / torch.sum(D2)
            cumprobs = torch.cumsum(probs, dim=0)
            init[i, :] = data[torch.searchsorted(cumprobs, r.sample([1]).to(data.device))]
    return init
class KMeansGPU:
  '''
  Kmeans clustering algorithm implemented with PyTorch

  Parameters:
    n_clusters: int, 
      Number of clusters

    max_iter: int, default: 100
      Maximum number of iterations

    tol: float, default: 0.0001
      Tolerance
    
    verbose: int, default: 0
      Verbosity

    mode: {'euclidean', 'cosine'}, default: 'euclidean'
      Type of distance measure
      
    init_method: {'random', 'point', '++'}
      Type of initialization

    minibatch: {None, int}, default: None
      Batch size of MinibatchKmeans algorithm
      if None perform full KMeans algorithm
      
  Attributes:
    centroids: torch.Tensor, shape: [n_clusters, n_features]
      cluster centroids
  '''
  def __init__(self, n_clusters, max_iter=200, tol=1e-4, verbose=0, mode="euclidean",device=torch.device("cuda:0")):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.mode = mode
    self.device=device
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    self.minibatch=int(33e6/self.n_clusters*info.free/ 1024 / 1024 / 1024)
    print("free_mem/GB:",info.free/ 1024 / 1024 / 1024,"minibatch:",self.minibatch)
    
  @staticmethod
  def cos_sim(a, b):
    """
      Compute cosine similarity of 2 sets of vectors

      Parameters:
      a: torch.Tensor, shape: [m, n_features]

      b: torch.Tensor, shape: [n, n_features]
    """
    return normalize(a, dim=-1) @ normalize(b, dim=-1).transpose(-2, -1)

  @staticmethod
  def euc_sim(a, b):
    """
      Compute euclidean similarity of 2 sets of vectors
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    return 2 * a @ b.transpose(-2, -1) -(a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]

  def max_sim(self, a, b):
    """
      Compute maximum similarity (or minimum distance) of each vector
      in a with all of the vectors in b
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    if self.mode == 'cosine':
      sim_func = self.cos_sim
    elif self.mode == 'euclidean':
      sim_func = self.euc_sim
    sim = sim_func(a, b)
    max_sim_v, max_sim_i = sim.max(dim=-1)
    return max_sim_v, max_sim_i

  def fit_predict(self, X):
    """
      Combination of fit() and predict() methods.
      This is faster than calling fit() and predict() seperately.
      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]
      centroids: {torch.Tensor, None}, default: None
        if given, centroids will be initialized with given tensor
        if None, centroids will be randomly chosen from X
      Return:
      labels: torch.Tensor, shape: [n_samples]

            mini_=33kk/k*remain
            mini=min(mini_,fea_shape)
            offset=log2(k/1000)*1.5
            kpp_all=min(mini_*10/offset,fea_shape)
            kpp_sample=min(mini_/12/offset,fea_shape)
    """
    assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
    assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
    assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "
    # print("verbose:%s"%self.verbose)

    offset = np.power(1.5,np.log(self.n_clusters / 1000))/np.log(2)
    with torch.no_grad():
      batch_size= X.shape[0]
      # print(self.minibatch, int(self.minibatch * 10 / offset), batch_size)
      start_time = time()
      if (self.minibatch*10//offset< batch_size):
        x = X[torch.randint(0, batch_size,[int(self.minibatch*10/offset)])].to(self.device)
      else:
        x = X.to(self.device)
      # print(x.device)
      self.centroids = _kpp(x, self.n_clusters, min(int(self.minibatch/12/offset),batch_size))
      del x
      torch.cuda.empty_cache()
      # self.centroids = self.centroids.to(self.device)
      num_points_in_clusters = torch.ones(self.n_clusters, device=self.device, dtype=X.dtype)#全1
      closest = None#[3098036]#int64
      if(self.minibatch>=batch_size//2 and self.minibatch<batch_size):
        X = X[torch.randint(0, batch_size,[self.minibatch])].to(self.device)
      elif(self.minibatch>=batch_size):
        X=X.to(self.device)
      for i in range(self.max_iter):
        iter_time = time()
        if self.minibatch<batch_size//2:#可用minibatch数太小，每次都得从内存倒腾到显存
          x = X[torch.randint(0, batch_size, [self.minibatch])].to(self.device)
        else:#否则直接全部缓存
          x = X

        closest = self.max_sim(a=x, b=self.centroids)[1].to(torch.int16)#[3098036]#int64#0~999
        matched_clusters, counts = closest.unique(return_counts=True)#int64#1k
        expanded_closest = closest[None].expand(self.n_clusters, -1)#[1000, 3098036]#int16#0~999
        mask = (expanded_closest==torch.arange(self.n_clusters, device=self.device)[:, None]).to(X.dtype)#==后者是int64*1000
        c_grad = mask @ x / mask.sum(-1)[..., :, None]
        c_grad[c_grad!=c_grad] = 0 # remove NaNs
        error = (c_grad - self.centroids).pow(2).sum()
        if self.minibatch is not None:
          lr = 1/num_points_in_clusters[:,None] * 0.9 + 0.1
        else:
          lr = 1
        matched_clusters=matched_clusters.long()
        num_points_in_clusters[matched_clusters] += counts#IndexError: tensors used as indices must be long, byte or bool tensors
        self.centroids = self.centroids * (1-lr) + c_grad * lr
        if self.verbose >= 2:
          print('iter:', i, 'error:', error.item(), 'time spent:', round(time()-iter_time, 4))
        if error <= self.tol:
          break

      if self.verbose >= 1:
        print(f'used {i+1} iterations ({round(time()-start_time, 4)}s) to cluster {batch_size} items into {self.n_clusters} clusters')
    return closest
