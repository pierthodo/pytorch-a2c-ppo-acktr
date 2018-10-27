import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space,num_processes,num_steps,N_backprop, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.N_backprop = N_backprop
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks,prev_value, deterministic=False):
        value, actor_features, rnn_hxs,beta_v = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        prev_value = masks * prev_value + (1 - masks) * value
        prev_value = beta_v * value + (1 - beta_v) * prev_value

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs,beta_v,prev_value

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def get_index(self,indices):
        del_index = [i*self.num_steps for i in list(range(self.num_processes))] ## First you identify the index of the end of the storage of the processes
        index_ext = []
        for b in indices:
            lim = del_index[int(b/(self.num_steps))]
            tmp = []
            for i in reversed(range(self.N_backprop)):
                if not b-i < lim: # if the index used goes on another process memory than block it
                    tmp.append(b-i)
            index_ext.append(tmp)
        return index_ext

    def evaluate_actions(self, inputs, rnn_hxs, masks, action,indices,rewards):
        #l = range(len(indices_ext))[self.N_backprop - 1::self.N_backprop] ## List of index for the original list

        _, actor_features, _,_ = self.base(inputs[indices], rnn_hxs[indices], masks[indices])
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action[indices])
        dist_entropy = dist.entropy().mean()

        indices_ext = self.get_index(indices)
        indices_ext_flat = [item for sublist in indices_ext for item in sublist]
        value, _, rnn_hxs,beta_v = self.base(inputs[indices_ext_flat], rnn_hxs[indices_ext_flat], masks[indices_ext_flat])

        value_mixed = []
        idx = 0
        for i in range(len(indices)):
            prev_value = value[idx]
            for p in indices_ext[i]:
                prev_value = masks[p] * prev_value + (1 - masks[p]) * value[idx]
                prev_value = beta_v[idx]* value[idx] + (1 - beta_v[idx]) * prev_value
                prev_value = prev_value - rewards[p]
                idx += 1
            value_mixed.append(prev_value+rewards[p])
        value_mixed = torch.stack(value_mixed, dim=0)

        return value_mixed, action_log_probs, dist_entropy, rnn_hxs, beta_v


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512,est_value=False,init_bias=0):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)
        self.est_value = est_value
        self.init_bias = init_bias
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic = init_(nn.Linear(hidden_size, 1))

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, self.init_bias))

        self.beta_net_value = nn.Sequential(
            init_(nn.Linear(hidden_size, 1)),
            nn.Sigmoid()
        )
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        if self.est_value:
            with torch.no_grad():
                hidden_value_beta = self.main(inputs / 255.0)
            beta_value = self.beta_net_value(hidden_value_beta)
        else:
            beta_value = torch.ones_like(masks)
        return self.critic(x), x, rnn_hxs,beta_value


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64,est_value=False,init_bias=0):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)
        self.est_value = est_value
        self.init_bias = init_bias
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))


        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, self.init_bias))

        self.beta_net_value = nn.Sequential(
            init_(nn.Linear(hidden_size, 1)),
            nn.Sigmoid()
        )

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        if self.est_value:
            with torch.no_grad():
                hidden_value_beta = self.critic(x)
            beta_value = self.beta_net_value(hidden_value_beta)
        else:
            beta_value = torch.ones_like(masks)
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs,beta_value
