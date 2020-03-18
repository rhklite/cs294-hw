import torch

import numpy as np
import print_custom as db
import unittest
import scipy


from train_pg_f18 import build_mlp
from train_pg_f18 import PolicyNet
from train_pg_f18 import Agent


class ModelTest(unittest.TestCase):

    # def testModelOutputShape(self):
    #     input_size, ouput_size, layers, hidden = 5, 5, 20, 10
    #     model = build_mlp(input_size, ouput_size, hidden, layers)

    #     batch = 5
    #     inputs, outputs = [batch, input_size], [batch, ouput_size]
    #     out = model(torch.randn(inputs))
    #     self.assertEqual(list(out.shape), outputs)

    def testSampleAction(self):
        ob_dim, ac_dim, n_layers, hidden_size = 5, 3, 20, 10
        batch = 1
        neural_network_args = {
            'n_layers': n_layers,
            'ob_dim': ob_dim,
            'ac_dim': ac_dim,
            'discrete': False,
            'size': hidden_size
        }
        network = PolicyNet(neural_network_args)

        inputs, outputs = [batch, ob_dim], [batch, ac_dim]
        # dist = torch.distributions.Categorical(action_probs)
        obs = torch.randn(inputs)
        if neural_network_args['discrete']:
            out = network(obs)
            action_probs = torch.nn.functional.softmax(out, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            sampled_action = dist.sample()
            dist.log_prob(sampled_action)

            dist1 = torch.distributions.Categorical(probs=action_probs)
            dist2 = torch.distributions.Categorical(logits=out)
            db.printInfo(dist1)
            db.printInfo(dist2)
        else:

            ts_mean, ts_logstd = network(obs)

            ts_mean = torch.randn(2000, 6)
            ts_logstd = torch.randn(6)

            dist = torch.distributions.Normal(loc=ts_mean, scale=ts_logstd)

            # YOUR_CODE_HERE
            ts_logstd_na = []
            for _ in range(list(ts_mean.shape)[0]):
                ts_logstd_na.append(ts_logstd)
            ts_logstd_na = torch.stack(ts_logstd_na)
            # db.printInfo(ts_logstd_na)
            # db.printInfo(ts_logstd_na.exp())

            sampled_action = torch.normal(mean=ts_mean, std=ts_logstd_na.exp())

            # sampled_action = torch.distributions.Normal(mean, logstd.exp()).sample()

            # sampled_action = torch.normal(mean, logstd.exp())

            # sampled_action = torch.normal(mean=torch.tensor([0,0]),
            #                               std=1)
        db.printInfo(sampled_action)
        db.printInfo(obs)
        db.printInfo()
        # db.printInfo(dist)
        # db.printInfo(dist.sample())
        # db.printInfo("Multinomial")
        # db.printInfo(torch.multinomial(action_probs, num_samples=1).view(-1))

        # db.printInfo(action)
        # self.assertEqual(list(out.shape), outputs)

    def testGetLogProb(self):
        agent = Agent(neural_network_args,)
        ob_dim, ac_dim, n_layers, hidden_size = 5, 3, 20, 10
        batch = 1
        neural_network_args = {
            'n_layers': n_layers,
            'ob_dim': ob_dim,
            'ac_dim': ac_dim,
            'discrete': False,
            'size': hidden_size
        }
        network = PolicyNet(neural_network_args)

        inputs, outputs = [batch, ob_dim], [batch, ac_dim]
        # dist = torch.distributions.Categorical(action_probs)
        obs = torch.randn(inputs)
        if neural_network_args['discrete']:
            out = network(obs)
            action_probs = torch.nn.functional.softmax(out, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            sampled_action = dist.sample()
            dist.log_prob(sampled_action)

            dist1 = torch.distributions.Categorical(probs=action_probs)
            dist2 = torch.distributions.Categorical(logits=out)
            db.printInfo(dist1)
            db.printInfo(dist2)
        else:

            ts_mean, ts_logstd = network(obs)

            # YOUR_CODE_HERE
            ts_logstd_na = []
            for _ in range(list(ts_mean.shape)[0]):
                ts_logstd_na.append(ts_logstd)
            ts_logstd_na = torch.stack(ts_logstd_na)
            # db.printInfo(ts_logstd_na)
            # db.printInfo(ts_logstd_na.exp())

            sampled_action = torch.normal(mean=ts_mean, std=ts_logstd_na.exp())

            # sampled_action = torch.distributions.Normal(mean, logstd.exp()).sample()

            # sampled_action = torch.normal(mean, logstd.exp())

            # sampled_action = torch.normal(mean=torch.tensor([0,0]),
            #                               std=1)
        db.printInfo(sampled_action)
        db.printInfo(obs)
        db.printInfo()
        # db.printInfo(dist)
        # db.printInfo(dist.sample())
        # db.printInfo("Multinomial")
        # db.printInfo(torch.multinomial(action_probs, num_samples=1).view(-1))

        # db.printInfo(action)
        # self.assertEqual(list(out.shape), outputs)


def rtg_solution(re_n, gamma, reward_to_go=True):
    db.printTensor(re_n)
    # YOUR_CODE_HERE
    if reward_to_go:
        q_n = [scipy.signal.lfilter(
            b=[1], a=[1, - gamma], x=re[::-1])[::-1] for re in re_n]
    else:
        q_n = [np.full_like(re, scipy.signal.lfilter(b=[1], a=[1, -gamma], x=re[::-1])[-1])
                for re in re_n]
    db.printInfo(q_n)
    q_n = np.concatenate(q_n).astype(np.float32)
    db.printInfo(q_n)
    return q_n


def rtg(re_n, gamma, reward_to_go=True):
    db.printTensor(re_n)
    q_n = []
    if reward_to_go:
        for traj in re_n:
            q_path = []
            for reward in traj[::-1]:
                try:
                    q_path.append(q_path[-1]*gamma + reward)
                except IndexError:
                    q_path.append(reward)
            q_n.append(q_path[::-1])
    else:
        # for traj in re_n:
        #     q_path = 0
        #     for t, reward in enumerate(traj[::-1]):
        #         q_path = q_path*gamma + reward
        #         db.printInfo(q_path)
        #         # db.printInfo(q_path)
        #         # db.printInfo(t)
        #         # db.printInfo(gamma**t)

        #         # input()
        #     # do this to have the same return for each time step
        #     q_n.append([q_path for _ in range(len(traj))])
        for traj in re_n:
            for t, reward in enumerate(traj):
                try:
                    q_path = q_path + reward*gamma**t
                except:
                    q_path = reward
            # do this to have the same return for each time step
            q_n.append([q_path for _ in range(len(traj))])
    db.printInfo(q_n)
    q_n = np.concatenate(q_n).astype(np.float32)

    db.printInfo(q_n)
    return q_n

def normalize(tensor, target_mean=0, target_std=1):
    """Shifts the input to have a target mean and target standard deviation.
    """
    return target_mean + (tensor - tensor.mean())*(target_std/(tensor.std() + 1e-7))

import torch.nn as nn

def mlp_sol(input_size, output_size, n_layers,  hidden_size, activation=nn.Tanh):
    layers = []
    for _ in range(n_layers):
        layers += [nn.Linear(input_size, hidden_size), activation()]
        input_size = hidden_size
    layers += [nn.Linear(hidden_size, output_size)]
    print(layers)

def mlp(input_size, output_size, n_layers, hidden_size,  activation=nn.Tanh):
    layers = []
    layers.append(nn.Linear(input_size,hidden_size))
    layers.append(activation())
    for _ in range(n_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation())
    layers.append(nn.Linear(hidden_size,output_size))
    print(layers)

if __name__ == "__main__":
    # unittest.main()

    re_n = np.arange(1,21).reshape(2,10)
    re_n= np.ones(5).reshape(1,5)
    re_n = np.arange(1,5).reshape(1,4)

    db.printInfo(re_n)
    # re_n = np.ones(10).reshape(1,10)
    q_n = rtg(re_n, 0.5, False)
    q_n_sol = rtg_solution(re_n,0.5, False)
    # q_n_sol = rtq_v2(re_n,0.5, False)

    # db.printInfo('Equal {}'.format(False not in (q_n == q_n_sol)))
    # mlp_sol(5,3,2,64)
    # mlp(5,3,2,64)
    # module = build_mlp(5,3,3,64)
    # print(module)