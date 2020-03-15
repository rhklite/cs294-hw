import torch

import print_custom as db
import unittest


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
            dist= torch.distributions.Categorical(action_probs)
            sampled_action = dist.sample()
            dist.log_prob(sampled_action)

            dist1= torch.distributions.Categorical(probs=action_probs)
            dist2 = torch.distributions.Categorical(logits=out)
            db.printInfo(dist1)
            db.printInfo(dist2)
        else:

            ts_mean, ts_logstd = network(obs)

            ts_mean = torch.randn(2000,6)
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
            dist= torch.distributions.Categorical(action_probs)
            sampled_action = dist.sample()
            dist.log_prob(sampled_action)

            dist1= torch.distributions.Categorical(probs=action_probs)
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

if __name__ == "__main__":
    unittest.main()
