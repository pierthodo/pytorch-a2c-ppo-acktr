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

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import xml.etree.ElementTree as ET
from gym.envs.registration import register
from shutil import copyfile

def make_mujoco_file(env_name,offline_directory,dt):
    game_list = {"Walker2d-v2":"walker2d",
                 "Hopper-v2":"hopper",
                 "Swimmer-v2":"swimmer",
                 "HalfCheetah-v2":"half_cheetah",
                 "InvertedDoublePendulum-v2":"inverted_double_pendulum"}
    copyfile('./gym/gym/envs/mujoco/assets/'+game_list[env_name]+ '.xml',offline_directory+game_list[env_name]+ '.xml')
    tree = ET.parse(offline_directory+game_list[env_name]+ '.xml')
    root = tree.getroot()

    if dt != 0:
        for i in root:
            if "timestep" in i.attrib.keys():
                i.attrib["timestep"] = str(float(i.attrib['timestep'])*dt)
    file = ET.tostring(root, encoding='utf8').decode('utf8')
    path = offline_directory + str.lower(env_name[:-3])+ '_tmp.xml'
    with open(path,'w') as f:
        f.write(file)
    return path

def main():
    args = get_args()
    path = make_mujoco_file(args.env_name,args.offline_directory,args.dt)


    if args.algo in ["a2c","acktr"]:
        raise "Not implemented"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device,False,path)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,args.num_processes,args.N_backprop,args.num_steps,args.recurrent_policy,args.N_recurrent,
        base_kwargs={'recurrent': args.recurrent_policy,'est_value': args.est_value})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            lr_beta=args.lr_beta,
            eps=args.eps,
            N_recurrent=args.N_recurrent,
            weighted_loss=args.weighted_loss,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    result = []
    result_val = []
    start = time.time()
    prev_value = 0
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, value_mixed, beta_v = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],prev_value)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            obs =  obs + torch.randn_like(obs) * args.noise_obs

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, value_mixed,beta_v)
            prev_value = value_mixed
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, "model"+ ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            result.append({"step":j * args.num_steps * args.num_processes,"mean reward": np.mean(episode_rewards),
                           "beta mean":np.array(rollouts.beta_v.data.cpu()).mean(),
                           "beta std":np.array(rollouts.beta_v.data.cpu()).std()})

        if j % 10 == 0 and len(episode_rewards) > 1:
            pickle.dump(result, open(args.offline_directory + "data.pkl", "wb"))
            pickle.dump(np.array(result_val),open(args.offline_directory + "val_data.pkl", "wb"))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            mean_reward_val = []
            for i in range(2):
                ob_rms = utils.get_vec_normalize(envs).ob_rms
                mean_reward_val.append(evaluate(actor_critic, ob_rms, args.env_name, i+10000,
                         args.num_processes, eval_log_dir, device,path))
            result_val.append({"step":j * args.num_steps * args.num_processes,"val reward":np.array(mean_reward_val).mean()})

if __name__ == "__main__":
    main()
