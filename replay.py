from collections import deque
import numpy as np
import random
import torch

class ReplayBuffer(object):
    def __init__(self, max_size=1000, seed=None):
        self.buffer = deque(maxlen=max_size)

    def add(self, s, a, r, s_, d):
        arr = [s, a, r, s_, d]
        self.buffer.append(arr)

    def add_(self, s, a, r, s_, d):
        self.buffer.append((s.detach().cpu().numpy(), a.detach().cpu().numpy(), r.detach().cpu().numpy(), s_.detach().cpu().numpy(), d))

    def sample(self, batch_size=100):
        ct = min(batch_size, len(self.buffer))
        # idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = random.sample(self.buffer, ct)
        # s = np.float32(x[0] for x in batch)
        # a = np.float32(x[1] for x in batch)
        # r = np.float32(x[2] for x in batch)
        # s_ = np.float32(x[3] for x in batch)
        # d = np.float32(x[4] for x in batch)
        s = [x[0] for x in batch]
        a = [x[1] for x in batch]
        r = [x[2] for x in batch]
        s_ = [x[3] for x in batch]
        d = [x[4] for x in batch]

        # return s, a, r.reshape(-1, 1), s_, d.reshape(-1, 1)
        return np.asarray(s), a, r, s_, d


    def get_batch(self, batch_size=100):
        states = []
        actions = []
        rewards = []
        next_state = []
        done = []

        for i in range(batch_size):
            ct = min(batch_size, len(self.buffer))
            replay = random.choice(self.buffer)
            states.append(replay[0])
            actions.append(replay[1])
            rewards.append(replay[2])
            next_state.append(replay[3])
            done.append(replay[4])

        rewards = np.array(rewards)
        rewards = rewards.reshape((rewards.shape[0], 1))
        # return torch.Tensor(states), torch.Tensor(actions), torch.Tensor(rewards), torch.Tensor(next_state), torch.Tensor(done)
        return states, actions, rewards, next_state, done

    # def sample_(self, batch_size=100):
    #     ct = min(batch_size, len(self.buffer))

