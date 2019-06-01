import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space,num_processes=1,N_backprop=5,num_steps=5,recurrent_policy=0,N_recurrent=0, base=None, base_kwargs=None):
        self.N_backprop = N_backprop
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.recurrent_policy = recurrent_policy
        self.N_recurrent = N_recurrent
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
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

    def act(self, inputs, rnn_hxs, masks, prev_value, deterministic=False):
        value, actor_features, rnn_hxs, beta_v = self.base(inputs, rnn_hxs, masks)

        ## RECURRENT LEARNING ADDITION ###
        prev_value = masks * prev_value + (1 - masks) * value
        value_mixed = beta_v * value + (1 - beta_v) * prev_value
        ##
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs, value_mixed, beta_v

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
                if not b-i < 0: # if the index used goes on another process memory than block it
                    tmp.append(b-i)
            index_ext.append(tmp)
        return index_ext

    def evaluate_actions(self, inputs, rnn_hxs, masks, action,prev_value_list,indices):

        ## RECURRENT TD
        if self.recurrent_policy and self.N_recurrent == 0:
            value, actor_features, rnn_hxs,beta_v = self.base(inputs, rnn_hxs, masks)
            dist = self.dist(actor_features)

            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

            return value, action_log_probs, dist_entropy, rnn_hxs, beta_v

        elif self.recurrent_policy and self.N_recurrent > 0:
            #raise("TO BE IMPLEMENTED")
            indices_ext = self.get_index(indices)
            indices_ext_flat = [item for sublist in indices_ext for item in sublist]
            value_mixed = []
            actor_features_list = []

            for i in range(len(indices)):
                value,actor_features,_,beta_v = self.base(inputs[indices_ext[i]],rnn_hxs[indices_ext[i]],
                                                                masks[indices_ext[i]])

                prev_value = prev_value_list[indices_ext[i][0]]
                for idx,p in enumerate(indices_ext[i]):
                    prev_value = masks[p] * prev_value + (1 - masks[p]) * value[idx]
                    prev_value = beta_v[idx] * value[idx] + (1 - beta_v[idx]) * prev_value

                value_mixed.append(prev_value)
                actor_features_list.append(actor_features[-1])

            value_mixed = torch.stack(value_mixed,dim=0)
            actor_features = torch.stack(actor_features_list,dim=0)
            dist = self.dist(actor_features)
            action_log_probs = dist.log_probs(action[indices])
            dist_entropy = dist.entropy().mean()

            return value_mixed, action_log_probs, dist_entropy, rnn_hxs,beta_v


        else:
            value_original, actor_features, _, _ = self.base(inputs[indices], rnn_hxs[indices], masks[indices])
            dist = self.dist(actor_features)

            action_log_probs = dist.log_probs(action[indices])
            dist_entropy = dist.entropy().mean()

            indices_ext = self.get_index(indices)
            indices_ext_flat = [item for sublist in indices_ext for item in sublist]
            value, _, _, beta_v = self.base(inputs[indices_ext_flat], rnn_hxs[indices_ext_flat],
                                                  masks[indices_ext_flat])
            value_mixed = []
            mean_beta_v_list = []
            idx = 0
            for i in range(len(indices)):
                prev_value = prev_value_list[indices_ext[i][0]]
                mean_beta_v = []
                for p in indices_ext[i]:
                    prev_value = masks[p] * prev_value + (1 - masks[p]) * value[idx]
                    prev_value = beta_v[idx] * value[idx] + (1 - beta_v[idx]) * prev_value
                    mean_beta_v.append(beta_v[idx])
                    idx += 1

                mean_beta_v_list.append(torch.stack(mean_beta_v,dim=0).mean())
                value_mixed.append(prev_value)

            value_mixed = torch.stack(value_mixed, dim=0)
            mean_beta_v = torch.stack(mean_beta_v_list,dim=0).detach()
            #
            #value_mixed, _, _, _ = self.base(inputs[indices], rnn_hxs[indices], masks[indices])
            return value_mixed, action_log_probs, dist_entropy, rnn_hxs,mean_beta_v


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

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
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64,num_layers=2, est_value = False):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)
        self.est_value = est_value
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        if num_layers ==2:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        else:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        ## RECURRENT TD
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.beta_net_value = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.beta_net_value_linear = nn.Sequential(
            init_(nn.Linear(hidden_size, 1)),
            nn.Sigmoid()
        )
        ##


        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)


        ### RECURRENT TD
        if self.est_value:
            hidden_value_beta = self.beta_net_value(x)
            beta_value = self.beta_net_value_linear(hidden_value_beta)
        else:
            beta_value = torch.ones_like(masks)
        ##


        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs, beta_value
