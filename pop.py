import multiprocessing as mp
import numpy as np
import torch as pt
from bipedal_walker import BipedalWalker
from modules import WorldModel, Controller

def random_rollout(env, seq_len=1600):
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  obs_data = np.zeros((seq_len+1, obs_dim), dtype=np.float32)
  act_data = np.zeros((seq_len, act_dim), dtype=np.float32)
  
  obs = env.reset()
  obs_data[0] = obs
  for t in range(seq_len):
    act = env.action_space.sample()
    obs, rew, done, _ = env.step(act)
    obs_data[t+1] = obs
    act_data[t] = act
    if done:
      obs = env.reset()

  return obs_data, act_data

def rollout(env, rnn, ctrl, seq_len=1600, render=False):
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  obs_data = np.zeros((seq_len+1, obs_dim), dtype=np.float32)
  act_data = np.zeros((seq_len, act_dim), dtype=np.float32)
  
  obs = env.reset()
  hid = (pt.zeros(1, rnn.hid_dim), # h
         pt.zeros(1, rnn.hid_dim)) # c

  obs_data[0] = obs
  for t in range(seq_len):
    if render:
      env.render()
    obs = pt.from_numpy(obs).unsqueeze(0)
    with pt.no_grad():
      act = ctrl(obs, hid[0])
      _, _, hid = rnn(obs, act, hid)

    act = act.squeeze().numpy()
    obs, rew, done, _ = env.step(act)
    obs_data[t+1] = obs
    act_data[t] = act
    if done:
      obs = env.reset()

  return obs_data, act_data

def evaluate(env, rnn, ctrl, num_episodes=5, max_episode_steps=1600):
  fitness = 0.0
 
  for ep in range(num_episodes):
    # Initialize observation and hidden states.
    obs = env.reset()
    hid = (pt.zeros(1, rnn.hid_dim), # h
           pt.zeros(1, rnn.hid_dim)) # c

    for t in range(max_episode_steps):
      obs = pt.from_numpy(obs).unsqueeze(0)
      with pt.no_grad():
        # Take an action with the controller.
        act = ctrl(obs, hid[0])

        # Predict the next observation with the RNN.
        _, _, hid = rnn(obs, act, hid)

      # Take a step in the environment.
      act = act.squeeze().numpy()
      obs, rew, done, _ = env.step(act)

      fitness += rew
      if done:
        break

  return fitness / num_episodes

class Population:
  def __init__(self, num_workers, agents_per_worker):
    self.num_workers = num_workers
    self.agents_per_worker = agents_per_worker
    self.popsize = num_workers * agents_per_worker

    self.pipes = []
    self.procs = []
    for rank in range(num_workers):
      parent_pipe, child_pipe = mp.Pipe()
      proc = mp.Process(target=self._worker,
                        name=f'Worker-{rank}', 
                        args=(rank, child_pipe, parent_pipe))
      self.pipes.append(parent_pipe)
      self.procs.append(proc)
      proc.daemon = True
      proc.start()
      child_pipe.close()

  def _worker(self, rank, pipe, parent_pipe):
    parent_pipe.close()

    rng = np.random.RandomState(rank)

    env = BipedalWalker()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    rnn = WorldModel(obs_dim, act_dim)
    ctrls = [Controller(obs_dim+rnn.hid_dim, act_dim)
             for _ in range(self.agents_per_worker)]
  
    while True:
      command, data = pipe.recv()

      if command == 'upload_rnn': # data: rnn
        rnn.load_state_dict(data.state_dict())
        pipe.send((None, True))

      elif command == 'upload_ctrl': # data: ([inds], noisy)
        inds, noisy = data
        for ctrl, ind in zip(ctrls, inds):
          if noisy:
            ind += rng.normal(scale=1e-3, size=ind.shape)
          ctrl.load_genotype(ind)
        pipe.send((None, True))

      elif command == 'rollout': # data: random_policy
        rollouts = []
        for ctrl in ctrls:
          env.seed(rng.randint(2**31-1))
          if data: # if rollout with random policy
            trajectory = random_rollout(env)
          else:
            trajectory = rollout(env, rnn, ctrl)
          rollouts.append(trajectory)
        pipe.send((rollouts, True))

      elif command == 'evaluate': # data: None
        evaluations = []
        for ctrl in ctrls:
          env.seed(rng.randint(2**31-1))
          evaluations.append(evaluate(env, rnn, ctrl))
        pipe.send((evaluations, True))

      elif command == 'close': # data: None
        env.close()
        pipe.send((None, True))
        return True

    return False

  def upload_rnn(self, rnn):
    for p in self.pipes:
      p.send(('upload_rnn', rnn))
    _, success = zip(*[p.recv() for p in self.pipes])
    return all(success)

  def upload_ctrl(self, ctrl, noisy=False):
    if isinstance(ctrl, np.ndarray):
      for p in self.pipes:
        inds = [np.copy(ctrl) for _ in range(self.agents_per_worker)]
        p.send(('upload_ctrl', (inds, noisy)))
    elif isinstance(ctrl, list):
      start = 0
      for p in self.pipes:
        end = start + self.agents_per_worker
        inds = [np.copy(c) for c in ctrl[start:end]]
        p.send(('upload_ctrl', (inds, noisy)))
        start = end
    else:
      return False

    _, success = zip(*[p.recv() for p in self.pipes])
    return all(success)

  def rollout(self, random_policy):
    for p in self.pipes:
      p.send(('rollout', random_policy))

    rollouts = []
    all_success = True
    for rollout, success in [p.recv() for p in self.pipes]:
      rollouts.extend(rollout)
      all_success = all_success and success 

    obs_batch = []
    act_batch = []
    for obs, act in rollouts:
      obs_batch.append(obs)
      act_batch.append(act)

    # (seq_len, batch_size, dim)
    obs_batch = pt.from_numpy(np.stack(obs_batch, axis=1))
    act_batch = pt.from_numpy(np.stack(act_batch, axis=1))
    return (obs_batch, act_batch), all_success

  def evaluate(self):
    for p in self.pipes:
      p.send(('evaluate', None))

    fits = []
    all_success = True
    for fit, success in [p.recv() for p in self.pipes]:
      fits.extend(fit)
      all_success = all_success and success

    return fits, all_success

  def close(self):
    for p in self.pipes:
      p.send(('close', None))
    _, success = zip(*[p.recv() for p in self.pipes])
    return all(success)
