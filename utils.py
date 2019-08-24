import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd

class ValueLogger:
  def __init__(self, name, bufsize=100):
    self.name = name
    self.bufsize = bufsize
    self._buffer = np.zeros((bufsize, 2))
    self._i = 0 # local iterator
    self._t = 0 # global iterator
    with open(f'{name}.csv', 'w') as f:
      f.write('step,value\n')

  def push(self, v):
    self._buffer[self._i] = (self._t, v)
    self._i += 1
    self._t += 1
    if self._i == self.bufsize:
      with open(f'{self.name}.csv', 'a') as f:
        for step, value in self._buffer:
          f.write(f'{step},{value}\n')
      self._buffer.fill(0)
      self._i = 0

  def plot(self, title, xlabel, ylabel):
    dat = pd.read_csv(f'{self.name}.csv')
    steps = dat['step']
    values = dat['value']
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(steps, values)
    plt.savefig(f'{self.name}.png')
    plt.close(fig=fig)
