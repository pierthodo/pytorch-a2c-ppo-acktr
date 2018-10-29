import torch
import torch.nn as nn
import torch.optim as optim

from .kfac import KFACOptimizer


class A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 lr_beta=0,
                 lr_value=0,
                 delib_coef=0,
                 delib_center=0.5,
                 acktr=False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.delib_coef = delib_coef
        self.delib_center = delib_center
        self.lr_beta = lr_beta

        self.bias_list = []
        self.param_list = []
        self.param_value = []
        for name, param in actor_critic.named_parameters():
            if "base.beta_net_value" in name:
                self.bias_list.append(param)
            elif "base.critic" in name:
                self.param_value.append(param)
            else:
                self.param_list.append(param)
        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                [{'params': self.param_list},
             {'params':self.param_value,'lr':lr_value},
             {'params': self.bias_list, 'lr': lr_beta}], lr, eps=eps, alpha=alpha)

    def update(self, rollouts): #### POSSIBLE TO SPEED UP A2C BY NOT REDOING THE ALL GRAPH TODO
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _, betas = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *rollouts.obs.size()[2:]), rollouts.recurrent_hidden_states[:-1].view(-1,
                                    rollouts.recurrent_hidden_states.size(-1)),
            rollouts.masks[:-1].view(-1, 1), rollouts.actions.view(-1, rollouts.actions.size(-1)),
            range(num_steps*num_processes), rollouts.rewards.view(-1, 1),rollouts.prev_value.view(-1,1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        if self.delib_coef > 0:
            target_beta = torch.zeros_like(betas).fill_(self.delib_center)
            delib_loss = F.mse_loss(betas, target_beta)
            delib_loss = delib_loss * value_loss
        else:
            delib_loss = torch.zeros_like(value_loss)


        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef+ delib_loss * self.delib_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item(),delib_loss.item()
