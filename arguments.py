import argparse

import torch



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--lr-value',type=float,default=-1)
    parser.add_argument('--name',type=str,default="default")
    parser.add_argument('--lr-beta',type=float,default=-1)
    parser.add_argument("--est-value", type=str2bool, nargs='?',
                        const=False, default="False",
                        help='estimate the beta for the value function')
    parser.add_argument('--gravity', type=float, default=-9.81)
    parser.add_argument('--init-bias',type=float, default=0,
                    help='Optimistic initalization for the beta value')
    parser.add_argument('--N-backprop',type=int,default=1,
                        help='Truncate backprop after n step')
    parser.add_argument('--disable-log', action='store_true', default=False)
    parser.add_argument('--delib-center', type=float, default=0,
            help='c in the || beta - c || ^ 2 loss')
    parser.add_argument('--delib-coef', type=float, default=0., 
            help='lambda in the lambda * || beta - c || ^ 2 loss')
    parser.add_argument('--beta-target', type=str2bool, nargs='?',
                        const=False, default="False",help="Use beta for the target")
    parser.add_argument('--noise-obs',type=float, default=0,
                    help='Noisy reward')
    parser.add_argument('--scatter',default=0,type=float, help='if > 0, value will determine max amt of episodes plotted')
    parser.add_argument('--beta-lambda',default=0,type=int,help="Use beta on the lambda target")
    parser.add_argument('--lr-bias',type=float,default=-1)
    parser.add_argument('--beta-fixed',type=float,default=1)
    parser.add_argument('--comet-offline',type=int,default=1)

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert (args.algo == 'ppo' or args.algo == 'a2c'), 'support for other agents not currently available'

    return args
