from comet_ml import Experiment
from comet_ml import OfflineExperiment
import copy
import glob
import os
import time
from collections import deque
import pickle
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import variation
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
from utils import get_vec_normalize,SampEn,beta_loss_series
from visualize import visdom_plot

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'
if args.lr_value == -1:
    args.lr_value = args.lr
if args.lr_beta == -1:
    args.lr_beta = args.lr
if args.lr_bias == -1:
    args.lr_bias = args.lr
if args.est_value == "False":
    args.N_backprop = 1
num_updates = int(args.num_frames) // args.num_steps // args.num_processes
gravity_list = [-9.81]
result = []
if args.comet_offline:
    experiment = OfflineExperiment(
                            project_name="estimate-value", workspace="pierthodo",disabled=args.disable_log,
                            log_code=False,auto_output_logging=None,  \
                            log_graph=False, auto_param_logging=False,parse_args=True, \
                            log_git_metadata=False,offline_directory=args.offline_directory)
else:
    experiment = Experiment(api_key="HFFoR5WtTjoHuBGq6lYaZhG0c",
                            project_name="estimate-value", workspace="pierthodo",disabled=args.disable_log,
                            log_code=False,auto_output_logging=None,  \
                            log_graph=False, auto_param_logging=False,parse_args=True, \
                            log_git_metadata=False, \
                            log_git_patch=False)
experiment.log_parameters(vars(args))
result.append(vars(args))
if args.tag_comet != "":
    experiment.add_tag(args.tag_comet)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,args.num_processes,args.num_steps,args.N_backprop,args.sub_reward,
        base_kwargs={'recurrent': args.recurrent_policy,'est_value':args.est_value,'init_bias':args.init_bias,'beta_fixed':args.beta_fixed,"share_beta":args.share_beta})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm , delib_coef=args.delib_coef, delib_center=args.delib_center,
                                lr_beta=args.lr_beta,lr_value=args.lr_value,)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, delib_coef=args.delib_coef, delib_center=args.delib_center, 
                         lr=args.lr,lr_beta=args.lr_beta,lr_value=args.lr_value,lr_bias=args.lr_bias,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size,args.sub_reward)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    prev_value = torch.zeros((rollouts.masks.size()[1],1))
    prev_value = prev_value.to(device)
    done = True
    episode_rewards = deque(maxlen=10)
    cum_reward = 0
    rollouts.masks[0] = 0
    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states,beta_v,new_prev_value = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],prev_value)
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            tmp = np.random.normal(obs.numpy(), scale=args.noise_obs)
            obs = torch.from_numpy(tmp)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            tmp = np.random.normal(reward.data[0][0],scale = 0)
            #tmp =1
            reward = torch.ones_like(reward) * tmp
            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            if args.beta_target:
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, new_prev_value, reward, masks,beta_v,new_prev_value)
            else:
                rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks,beta_v,new_prev_value)

            reward = reward.to(device)
            if args.sub_reward:
                prev_value = new_prev_value - reward
            else:
                prev_value = new_prev_value

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau,args.beta_lambda)

        value_loss, action_loss, dist_entropy, delib_loss = agent.update(rollouts)

        rollouts.after_update()
        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
        cum_reward += np.mean(episode_rewards)
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))
            prev_numpy = np.array(rollouts.prev_value[1:].data.squeeze())
            return_numpy = np.array(rollouts.returns.data.squeeze())
            value_numpy =  np.array(rollouts.value_preds.data.squeeze())
            #beta_loss_s= beta_loss_series(np.array(rollouts.prev_value.view(-1,1)[:-1,:].data),
            #                                    np.array(rollouts.value_preds.view(-1,1)[:-1,:].data),
            #                                    np.array(rollouts.returns.view(-1,1)[:-1,:].data),
            #                                    np.array(rollouts.beta_v.view(-1,1).data))
            #beta_loss_s = beta_loss_s.sum()
            # Calculate the loss with and without beta 
            loss_v = np.mean((value_numpy - return_numpy)**2)
            loss_v_tilde =  np.mean((prev_numpy - return_numpy[:-1])**2)
            #loss_v_mean = np.mean((prev_numpy - rollouts)**2) TODO 
            #


            prev_value_np = np.array(rollouts.prev_value.data)
            experiment.log_metrics({"mean reward": np.mean(episode_rewards),
                                             "Value loss": value_loss, "Action Loss": action_loss,
                                             "beta_v mean": np.array(rollouts.beta_v.data).mean(),
                                             "beta_v std": np.array(rollouts.beta_v.data).std(),
                                             "value mean": prev_value_np.mean(),"value std":prev_value_np.std(),
                                             "variation value":np.abs(variation(prev_value_np)[0][0]),"variance step value": np.abs(prev_value_np[1:]-prev_value_np[:-1]).mean(),
                                             "Error target - v":loss_v,"Error target - v_tilde":loss_v_tilde
                                             },

                                            step=j * args.num_steps * args.num_processes)
            result.append({"step":j * args.num_steps * args.num_processes,"mean reward": np.mean(episode_rewards),
                                             "Value loss": value_loss, "Action Loss": action_loss,
                                             "beta_v mean": np.array(rollouts.beta_v.data).mean(),
                                             "beta_v std": np.array(rollouts.beta_v.data).std(),
                                             "value mean": prev_value_np.mean(),"value std":prev_value_np.std(),
                                             "variation value":np.abs(variation(prev_value_np)[0][0]),"variance step value": np.abs(prev_value_np[1:]-prev_value_np[:-1]).mean(),
                                             "Error target - v":loss_v,"Error target - v_tilde":loss_v_tilde
                                             })
            if args.scatter:
                reward = np.array(rollouts.rewards.data)
                # find end of episodes
                is_done = rollouts.masks.cpu().data.numpy()
                is_done = np.where(is_done == 0)[0]
                    
                plt.ylim(0,1)
                for value in is_done:
                    x = 1 #plt.axvline(x=value, linestyle='--')
              
                bound = int(min(args.scatter, is_done.shape[0])) + 1
                colors = plt.cm.get_cmap('hsv', bound)
                colors = [colors(i) for i in range(bound)]

                for i, value in enumerate(is_done[:-1]):
                    if i >= args.scatter: 
                        break

                    rew_v  = rollouts.rewards.data[is_done[i]:is_done[i+1]].squeeze()
                    rew    = reward[is_done[i]:is_done[i+1]].squeeze()
                    beta_v = rollouts.beta_v.data[is_done[i]:is_done[i+1]]
                    beta   = beta_v.cpu().data.numpy().squeeze()
                    value  = rollouts.value_preds[is_done[i]:is_done[i+1]].squeeze()
                    target = rollouts.returns.data[is_done[i]:is_done[i+1]].squeeze()

                    # loop over the betas and the values to valculate \tilde{v}
                    value_tilde = rollouts.prev_value[is_done[i]:is_done[i+1]]

                    # we also want to plot the value using a fixed beta
                    beta_mean = beta_v.mean()
                    value_mean = []
                    prev_value_scat = value[0]
                    for ind in range(len(value) - 1):
                        v = value[ind + 1]
                        value_mean_t = beta_mean  * v + (1 - beta_mean) * prev_value_scat
                        prev_value_scat   = value_mean_t - rew_v[ind + 1]
                        value_mean  += [value_mean_t]

                    value_mean = torch.stack(value_mean)


                    """ First graph : value_tilde vs value vs beta """
                    fig, ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    ax1.set_ylim(0, 1)
                    
                    # for visibility, let's scaler the values to the 0, 1 interval
                    max_val = max(value.max(), value_tilde.max())
                    min_val = min(value.min(), value_tilde.min())
                    scale = lambda x : (x - min_val) / (max_val - min_val)
                    scaled_value = scale(value).squeeze().cpu().data.numpy()[1:]
                    scaled_value_tilde = scale(value_tilde).squeeze().cpu().data.numpy()[1:]
                    scaled_value_mean  = scale(value_mean).cpu().data.numpy()
                    scaled_target = scale(target).cpu().data.numpy()

                    ax2.plot(np.arange(scaled_value.shape[0]), scaled_value, label='V')
                    ax2.plot(np.arange(scaled_value.shape[0]), scaled_value_tilde, label='V~')
                    #ax2.plot(np.arange(scaled_value.shape[0]), scaled_value_mean, label='V_mean')
                    ax2.plot(np.arange(beta.shape[0]), beta, label='beta')
                    ax2.plot(np.arange(scaled_target.shape[0]), scaled_target, label='target')
                    ax2.legend()
                    experiment.log_figure( figure_name='V vs V~' + str(i)+"_"+str(j * args.num_steps * args.num_processes), figure=None)
                    plt.clf()


                    """ Second graph : beta vs reward """
                    fig, ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    ax1.set_ylim(0, 1)

                    #plt.plot(np.arange(rew.shape[0]), rew, c=colors[i], linestyle='dashed', s=8) #, marker='<')
                    ax1.plot(np.arange(beta.shape[0]), beta, c='b', label='beta')#, s=8, marker='+')
                    ax2.plot(np.arange(rew.shape[0]), rew, c='r', label='reward')#, s=8, marker='+')
                    ax2.legend()

                    ax2.scatter(np.arange(rew.shape[0]), rew, c=colors[i], s=12, marker='<')
                    ax1.scatter(np.arange(beta.shape[0]), beta, c='k', s=12, marker='>')
                    experiment.log_figure( figure_name='beta vs reward ' + str(i)+"_"+str(j * args.num_steps * args.num_processes), figure=None)
                    plt.clf()

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            for grav in gravity_list:
                eval_envs = make_vec_envs(
                    args.env_name, args.seed + args.num_processes, args.num_processes,
                    args.gamma, eval_log_dir, args.add_timestep, device, True)
                eval_envs.venv.venv.envs[0].env.env.model.opt.gravity[-1] = grav

                vec_norm = get_vec_normalize(eval_envs)
                if vec_norm is not None:
                    vec_norm.eval()
                    vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

                eval_episode_rewards = []
                prev_value_eval = torch.zeros((rollouts.masks.size()[1], 1))
                prev_value_eval = prev_value_eval.to(device)

                obs = eval_envs.reset()
                eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                actor_critic.recurrent_hidden_state_size, device=device)
                eval_masks = torch.zeros(args.num_processes, 1, device=device)

                while len(eval_episode_rewards) < 10:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states, _, _ = actor_critic.act(
                            obs,
                            eval_recurrent_hidden_states,
                            eval_masks, prev_value_eval,deterministic=True)
                        #_, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        #    obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                    # Obser reward and next obs
                    obs, reward, done, infos = eval_envs.step(action)

                    eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                    for done_ in done])
                    for info in infos:
                        if 'episode' in info.keys():
                            eval_episode_rewards.append(info['episode']['r'])
                            eval_envs = make_vec_envs(
                                args.env_name, args.seed + args.num_processes + len(eval_episode_rewards)
                                , args.num_processes,
                                args.gamma, eval_log_dir, args.add_timestep, device, True)

                            vec_norm = get_vec_normalize(eval_envs)
                            if vec_norm is not None:
                                vec_norm.eval()
                                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms
                            obs = eval_envs.reset()

                eval_envs.close()

                print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                    format(len(eval_episode_rewards),
                           np.mean(eval_episode_rewards)))
                experiment.log_metrics({"Gravity: "+str(grav)+"Eval mean reward":np.mean(eval_episode_rewards)},
                                                step=j * args.num_steps * args.num_processes)
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass
    pickle.dump(result,open(args.offline_directory+"data.pkl",'w'))

if __name__ == "__main__":
    main()
