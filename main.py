# Implementation of World Models (Ha & Schmidhuber, 2018).

import multiprocessing as mp
import numpy as np
import torch as pt
from torch import optim, distributions
from tqdm import tqdm
from es import EvolutionStrategy
from bipedal_walker import BipedalWalker
from modules import WorldModel, Controller
from pop import Population, rollout
from utils import ValueLogger

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

def train_rnn(rnn, optimizer, pop, random_policy=False, 
    num_rollouts=1000, filename='ha_rnn.pt', logger=None):
  rnn = rnn.train().to(device)

  batch_size = pop.popsize
  num_batch = num_rollouts // batch_size

  batch_pbar = tqdm(range(num_batch))
  for i in batch_pbar:
    # sample rollout data
    (obs_batch, act_batch), success = pop.rollout(random_policy)
    assert success

    obs_batch = obs_batch.to(device)
    act_batch = act_batch.to(device)

    obs_batch, next_obs_batch = obs_batch[:-1], obs_batch[1:]
    hid = (pt.zeros(batch_size, rnn.hid_dim).to(device),
           pt.zeros(batch_size, rnn.hid_dim).to(device))
    rnn.zero_grad()

    # compute NLL loss
    loss = 0.0
    for obs, act, next_obs in zip(obs_batch, act_batch, next_obs_batch):
      mu, sigma, hid = rnn(obs, act, hid)
      dist = distributions.Normal(loc=mu, scale=sigma)
      nll = -dist.log_prob(next_obs) # negative log-likelihood
      nll = pt.mean(nll, dim=-1)     # mean over dimensions
      nll = pt.mean(nll, dim=0)      # mean over batch
      loss += nll
    loss = loss / len(act_batch)     # mean over trajectory
    batch_pbar.set_description(f'loss={loss.item():.3f}')

    # update RNN
    loss.backward()
    optimizer.step()

    if logger is not None:
      logger.push(loss.item())

  pt.save(rnn.state_dict(), filename)

def evolve_ctrl(ctrl, es, pop, num_gen=100, filename='ha_ctrl.pt', logger=None):
  best_sol = None
  best_fit = -np.inf

  gen_pbar = tqdm(range(num_gen))
  for g in gen_pbar:
    # upload individuals
    inds = es.ask()
    success = pop.upload_ctrl(inds)
    assert success

    # evaluate
    fits, success = pop.evaluate()
    assert success
    
    # update
    es.tell(fits)
    best_sol, best_fit = es.best
    gen_pbar.set_description(f'best={best_fit:.3f}')

    if logger is not None:
      logger.push(best_fit)

  ctrl.load_genotype(best_sol)
  pt.save(ctrl.state_dict(), filename)

def main(args):
  print("IT'S DANGEROUS TO GO ALONE! TAKE THIS.")
  
  np.random.seed(0)
  pt.manual_seed(0)

  env = BipedalWalker()
  env.seed(0)

  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  print(f"Initializing agent (device={device})...")
  rnn = WorldModel(obs_dim, act_dim)
  ctrl = Controller(obs_dim+rnn.hid_dim, act_dim)

  # Adjust population size based on the number of available CPUs.
  num_workers = mp.cpu_count() if args.nproc is None else args.nproc
  num_workers = min(num_workers, mp.cpu_count())
  agents_per_worker = args.popsize // num_workers
  popsize = num_workers * agents_per_worker

  print(f"Initializing population with {popsize} workers...")
  pop = Population(num_workers, agents_per_worker)
  global_mu = np.zeros_like(ctrl.genotype)

  loss_logger = ValueLogger('ha_rnn_loss', bufsize=20)
  best_logger = ValueLogger('ha_ctrl_best', bufsize=100)

  # Train the RNN with random policies.
  print(f"Training M model with a random policy...")
  optimizer = optim.Adam(rnn.parameters(), lr=args.lr)
  train_rnn(rnn, optimizer, pop, random_policy=True, 
    num_rollouts=args.num_rollouts, logger=loss_logger)
  loss_logger.plot('M model training loss', 'step', 'loss')
  
  # Upload the trained RNN.
  success = pop.upload_rnn(rnn.cpu())
  assert success

  # Iteratively update controller and RNN.
  for i in range(args.niter):
    # Evolve controllers with the trained RNN.
    print(f"Iter. {i}: Evolving C model...")
    es = EvolutionStrategy(global_mu, args.sigma0, popsize)
    evolve_ctrl(ctrl, es, pop, num_gen=args.num_gen, logger=best_logger)
    best_logger.plot('C model evolution', 'gen', 'fitness')

    # Update the global best individual and upload them.
    global_mu = np.copy(ctrl.genotype)
    success = pop.upload_ctrl(global_mu, noisy=True)
    assert success
    
    # Train the RNN with the current best controller.
    print(f"Iter. {i}: Training M model...")
    train_rnn(rnn, optimizer, pop, random_policy=False,
      num_rollouts=args.num_rollouts, logger=loss_logger)
    loss_logger.plot('M model training loss', 'step', 'loss')

    # Upload the trained RNN.
    success = pop.upload_rnn(rnn.cpu())
    assert success

    # Test run!
    rollout(env, rnn, ctrl, render=True)

  success = pop.close()
  assert success

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--niter', type=int, default=10)
  parser.add_argument('--nproc', type=int, default=None)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--popsize', type=int, default=50)
  parser.add_argument('--sigma0', type=float, default=0.1)
  parser.add_argument('--num-gen', type=int, default=100)
  parser.add_argument('--num-rollouts', type=int, default=1000)
  args = parser.parse_args()

  main(args)
