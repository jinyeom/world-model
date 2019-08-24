import numpy as np
from cma import CMAEvolutionStrategy

class EvolutionStrategy:
  # Wrapper for CMAEvolutionStrategy
  def __init__(self, mu, sigma, popsize, weight_decay=0.01):
    self.es = CMAEvolutionStrategy(mu.tolist(), sigma, {'popsize': popsize})
    self.weight_decay = weight_decay
    self.solutions = None

  @property
  def best(self):
    best_sol = self.es.result[0]
    best_fit = -self.es.result[1]
    return best_sol, best_fit

  def _compute_weight_decay(self, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -self.weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

  def ask(self):
    self.solutions = self.es.ask()
    return self.solutions

  def tell(self, reward_table_result):
    reward_table = -np.array(reward_table_result)
    if self.weight_decay > 0:
      l2_decay = self._compute_weight_decay(self.solutions)
      reward_table += l2_decay
    self.es.tell(self.solutions, reward_table.tolist())
