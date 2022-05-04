""" 
Code from https://github.com/Kaixhin/PlaNet/blob/master/memory.py
"""
import torch


def preprocess_observation_(observation, bit_depth):
    """Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])"""
    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 **
                                                        bit_depth).sub_(0.5)
    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth)

                    
def postprocess_observation(observation, bit_depth):
    """Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])"""
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


class ExperienceReplay():

  def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device):
    self.device = device
    self.symbolic_env = symbolic_env
    self.size = size
    self.observations = np.empty((size, observation_size) if symbolic_env else (
        size, 3, 64, 64), dtype=np.float32 if symbolic_env else np.uint8)
    self.actions = np.empty((size, action_size), dtype=np.float32)
    self.rewards = np.empty((size, ), dtype=np.float32)
    self.nonterminals = np.empty((size, 1), dtype=np.float32)
    self.idx = 0
    self.full = False  # Tracks if memory has been filled/all slots are valid
    # Tracks how much experience has been used in total
    self.steps, self.episodes = 0, 0
    self.bit_depth = bit_depth

  def append(self, observation, action, reward, done):
    if self.symbolic_env:
      self.observations[self.idx] = observation
    else:
      # Decentre and discretise visual observations (to save memory)
      self.observations[self.idx] = postprocess_observation(
          observation, self.bit_depth)
    self.actions[self.idx] = action
    self.rewards[self.idx] = reward
    self.nonterminals[self.idx] = not done
    self.idx = (self.idx + 1) % self.size
    self.full = self.full or self.idx == 0
    self.steps, self.episodes = self.steps + \
        1, self.episodes + (1 if done else 0)

  # Returns an index for a valid single sequence chunk uniformly sampled from the memory
  def _sample_idx(self, L):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.size if self.full else self.idx - L)
      idxs = np.arange(idx, idx + L) % self.size
      # Make sure data does not cross the memory index
      valid_idx = not self.idx in idxs[1:]
    return idxs

  def _retrieve_batch(self, idxs, n, L):
    vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
    observations = torch.as_tensor(
        self.observations[vec_idxs].astype(np.float32))
    if not self.symbolic_env:
      # Undo discretisation for visual observations
      preprocess_observation_(observations, self.bit_depth)
    return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)

  # Returns a batch of sequence chunks uniformly sampled from the memory
  def sample(self, n, L):
    batch = self._retrieve_batch(np.asarray(
        [self._sample_idx(L) for _ in range(n)]), n, L)
    return [torch.as_tensor(item).to(device=self.device) for item in batch]