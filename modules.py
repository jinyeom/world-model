import torch as pt
from torch import nn, optim, distributions
from torch.nn import functional as F

class WorldModel(nn.Module):
  def __init__(self, obs_dim, act_dim, hid_dim=64):
    super(WorldModel, self).__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim 
    self.hid_dim = hid_dim

    self.lstm = nn.LSTMCell(obs_dim+act_dim, hid_dim)
    self.mu = nn.Linear(hid_dim, obs_dim)
    self.logsigma = nn.Linear(hid_dim, obs_dim)

  def forward(self, obs, act, hid):
    x = pt.cat([obs, act], dim=-1)
    h, c = self.lstm(x, hid)
    mu = self.mu(h)
    sigma = pt.exp(self.logsigma(h))
    return mu, sigma, (h, c)

class Phenotype(nn.Module):
  @property
  def genotype(self):
    params = [p.detach().view(-1) for p in self.parameters()]
    return pt.cat(params, dim=0).cpu().numpy()

  def load_genotype(self, params):
    start = 0
    for p in self.parameters():
      end = start + p.numel()
      new_p = pt.from_numpy(params[start:end])
      p.data.copy_(new_p.view(p.shape).to(p.device))
      start = end

class Controller(Phenotype, nn.Linear):
  def forward(self, obs, h):
    state = pt.cat([obs, h], dim=-1)
    return pt.tanh(super().forward(state))