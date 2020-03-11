"""Data Aggregatopm with behaviour cloning.


"""
import gym
import pickle
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import load_policy
import print_custom as db

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

# dev = 'cpu'

db.printInfo(dev)


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc4 = nn.Linear(hidden_size, output_dim)

    def forward(self, inputs):
        hidden = (torch.zeros(1, len(inputs), self.hidden_size, dtype=torch.double).to(dev),
                  torch.zeros(1, len(inputs), self.hidden_size, dtype=torch.double).to(dev))
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x, hidden = self.lstm(inputs.view(len(inputs), 1, -1), hidden)
        x = self.fc4(x[:,-1,:])

        return x.view(len(inputs), 1, -1)


def load_rollout(path):
    pickle_in = open(path, 'rb')
    rollout = pickle.load(pickle_in)
    return rollout


def feed_forward_generator(train, target, batch_size=10):
    """Random samples from the training set without replacement.

    Args:
        target (tensor): Training observations.
        train (tensor): Action outputs.
        batch_size (int, optional): Number of batches. Defaults to 3.

    Yields:
        [type]: [description]
    """
    n_samples = len(target)//batch_size
    perm = torch.randperm(len(target))
    for start_idx in range(0, len(target)-n_samples, n_samples):
        yield train[perm[start_idx:start_idx+n_samples]], target[perm[start_idx:start_idx+n_samples]]


def recurrent_generator(train, target, batch_size=10):
    assert len(
        target) > batch_size, 'Batch size need to be smaller then number of samples.'
    n_samples = len(target)//batch_size
    perm = torch.randperm(batch_size)
    for idx in range(batch_size):
        start_idx = perm[idx]*n_samples
        yield train[start_idx:start_idx+n_samples], target[start_idx:start_idx+n_samples]

def evaluate(obs, policy_net, env, max_timesteps=1000, render=False):
    returns = []
    actions = []
    for i in range(iteration):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            with torch.no_grad():
                action = policy_net(obs[None, :])
                action = action.to('cpu').numpy()
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps >= max_timesteps:
                    break


def save(epoch, model, optimizer, loss, path, overwrite=False):

    if overwrite:
        rev = 1
        while os.path.exists(path):
            path = path[:-4]
            path = path+'_'+str(rev)+'.pth'
            rev += 1

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)

    db.printInfo('Model saved to {}'.format(path))


def main():
    parser = argparse.ArgumentParser(
        'Behaviour cloning using pre-trained expert rollouts.')
    parser.add_argument('--rollout_file', type=str,
                        default='expert_data/Humanoid-v2.pkl')
    parser.add_argument('--envname', type=str, default='Humanoid-v2')
    parser.add_argument('--max_timesteps', type=int, default=1000)
    parser.add_argument('--training_epochs', type=int, default=2000)
    parser.add_argument('--save_model', type=str, default='./DAgger_Humanoid_lstm-v2.pth')
    parser.add_argument('--render', type=bool, default=True)

    args = parser.parse_args()

    # load expert rollout and model
    rollout = load_rollout(args.rollout_file)
    train = torch.tensor(rollout['observations'], dtype=torch.double)
    target = torch.tensor(rollout['actions'], dtype=torch.double)

    policy_net = load_policy.load_policy(args.expert_policy_file)

    train = train.to(dev)
    target = target.to(dev)

    db.printTensor(train)
    db.printTensor(target)

    # make the environment
    env = gym.make(args.envname)

    # build model
    model = Model(input_dim=env.observation_space.shape[0],
                  output_dim=env.action_space.shape[0])
    model.double()
    model.to(torch.device(dev))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # train
    for i in range(args.dagger_epochs):
        for epoch in range(args.training_epochs):
            data_generator = recurrent_generator(train, target, batch_size=20)
            t_start = time.time()
            for train_sample, target_sample in data_generator:
                out = model(train_sample)
                loss = criterion(out, target_sample)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                db.printInfo('Epoch: {} Loss: {} Time: {}'.format(epoch, loss, time.time()-t_start))

        obs = env.reset()
        done = False

    save(epoch, model, optimizer, loss, args.save_model)


if __name__ == "__main__":
    main()
