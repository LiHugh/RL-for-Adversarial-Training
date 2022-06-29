import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# import utils
from replay import ReplayBuffer

from torch.distributions import Normal
def fc_init(size, init=None):
    init = init or size[0]
    var = 1./np.sqrt(init)
    n = Normal(0, var)
    return n.sample(size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BOUND_INIT = 0.005


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim, 400)
        self.fc1.weight.data = fc_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(400, 256)
        self.fc2.weight.data = fc_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(256, action_dim)
        self.fc3.weight.data.uniform_(-BOUND_INIT, BOUND_INIT)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        action = F.tanh(self.fc3(x))
        action = self.action_lim * action

        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim+action_dim, 400)
        self.fc1.weight.data = fc_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(400, 256)
        self.fc2.weight.data = fc_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(256, 1)
        self.fc3.weight.data.uniform_(-BOUND_INIT, BOUND_INIT)

    def forward(self, state, action):
        # print('state shape:', state.shape, action.shape, state)
        state, action = state.view(state.shape[0], -1), action.view(action.shape[0], -1)
        x = F.relu(self.fc1(torch.cat((state, action), dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class DDPG(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.actor = Actor(state_dim, action_dim, action_lim).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_lim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def choose_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state = state.view(1, -1)
        # return self.actor(state).cpu().data.numpy().flatten()
        return self.actor(state)

    def learn(self, replay_buffer, episodes, batch_size=100, gamma=0.99, tau=0.005, learn_freq=20):
        # for eps in range(episodes):
            # sample replay buffer
        # s, a, r, s_, d = replay_buffer.sample(batch_size)
        # print("sample state shape:", len(s), type(s), s)
        s, a, r, s_, d = replay_buffer.get_batch(batch_size)
        # s = s.numpy()
        # state = [np.array(x[0], dtype=np.float32) for x in s]
        # state = np.array(s)
        state = torch.FloatTensor(s).to(device)
        action = torch.FloatTensor(a).to(device)
        reward = torch.FloatTensor(r).to(device)
        next_state = torch.FloatTensor(s_).to(device)
        # done = torch.FloatTensor(1-d).to(device)
        done = None

        # compute the targe Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (gamma*target_Q).detach()

        # compute current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_loss = critic_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # # Compute actor loss and update
        # actor_loss = -self.critic(state, self.actor(state)).mean()
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

        # Update the target network of the actor and critic
        if episodes % learn_freq == 0:
            # print('update the target network')
            # Compute actor loss and update
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau*param.data + (1-tau)*target_param.data)

    def save(self, directory=None, filename=None):
        # torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        # torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.actor.state_dict(), 'ACTOR_MINIST_10.pt')
        torch.save(self.critic.state_dict(), 'CRITIC_MINIST_10.pt')

    def load(self, directory, filename):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

