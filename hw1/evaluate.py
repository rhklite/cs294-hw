import argparse
import numpy as np

import gym
import torch

from behaviour_clone import Model
import print_custom as db

def load_model(path):
    """Loads the model state dict.
    TODO Load other saved objects.
    
    Args:
        path (str): Path to save file.
    """
    save_file = torch.load(path)
    return save_file['model_state_dict']

def main():
    parser = argparse.ArgumentParser(
        'Behaviour cloning using pre-trained expert rollouts.')
    parser.add_argument('--save_file', type=str,
                        default='./BC_Ant-v2.pth')
    parser.add_argument('--envname', type=str, default='Ant-v2')
    parser.add_argument('--iter', type=int, default=20)
    parser.add_argument('--max_timesteps', type=int, default=1000)
    parser.add_argument('--render', type=bool, default=False)
    
    args = parser.parse_args()

    env = gym.make(args.envname)

    # build model
    model = Model(input_dim=env.observation_space.shape[0],
                  output_dim=env.action_space.shape[0])
    model.double()
    model.load_state_dict(load_model(args.save_file))

    returns = []
    for i in range(args.iter):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            with torch.no_grad():
                action = model(torch.tensor(obs, dtype=torch.double).unsqueeze(0))
                action = action.numpy()
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps >= args.max_timesteps:
                    break
        db.printInfo("Iter {} {}/{} Reward: {:.2f}".format(i, steps, args.max_timesteps, totalr))
        returns.append(totalr)

    db.printInfo("Mean return {}".format(np.mean(returns)))
    db.printInfo("Std of return {}".format(np.std(returns)))

if __name__ == "__main__":
    main()