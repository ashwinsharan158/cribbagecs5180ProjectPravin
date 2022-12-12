import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .DQNAgent import DQN


class NFSPPolicy(nn.Module):
    """
      Policy used by agent for supervised learning for NFSP algorithm.
    """

    def __init__(self, input_size, hidden_layer_size, output_size):
        super(NFSPPolicy, self).__init__()
        self.policyNet = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.policyNet(x)


class NFSPPolicyAgent(object):
    def __init__(self, parameters):
        self.params = parameters
        self.action_dim = parameters['action_dimension']
        self.obs_dim = parameters['observation_dimension']
        self.behavior_network = DQN(input_size=parameters['observation_dimension'],
                                    hidden_layer_size=parameters['hidden_layer_dimension'],
                                    output_size=parameters['action_dimension'])
        self.target_network = DQN(input_size=parameters['observation_dimension'],
                                  hidden_layer_size=parameters['hidden_layer_dimension'],
                                  output_size=parameters['action_dimension'])
        # For supervised learning
        self.policy = NFSPPolicy(input_size=parameters['observation_dimension'],
                                 hidden_layer_size=parameters['hidden_layer_dimension'],
                                 output_size=parameters['action_dimension'])
        self.gamma = 0.99
        self.optimizer = torch.optim.Adam(self.behavior_network.parameters(), lr=parameters['learning_rate'])
        self.device = "cpu"

    def get_action(self, obs, eps, hand):
        obs = self.toTensorOneHot(obs).view(1, -1)
        eta = 0.3
        best_response = False
        if np.random.random() > eta:
            with torch.no_grad():
                q_values = self.policy(obs.float())
                action = np.argmax(q_values[0])
        else:
            best_response = True
            eps = 0.3
            if np.random.random() < eps:
                action = np.random.choice(hand, 1)[0]
                return action.run_val, True
            else:
                with torch.no_grad():
                    q_values = self.behavior_network(obs.float())
        l = []
        i = 0
        for x in q_values[0]:
            l.append((i, x))
            i = i + 1
        sort_orders = sorted(l, key=lambda x: x[1], reverse=True)
        action = 0
        runVal = list(map(lambda x: x.run_val, hand))
        for x, y in sort_orders:
            if x in runVal:
                action = x
                break
        # action = q_values.multinomial(1).item()
        if action == 0:
            action = 1
        if action == 14:
            action = 13
        return action, best_response

    # OBS to one hot encoding of num classes
    def one_hot_encoding(self, obs, num):
        y = np.array([])
        for i in obs:
            x = np.zeros(num)
            x[i] = 1
            y = np.concatenate((y, x), axis=None)
        return y

    def toTensorOneHot(self, obs):
        state = F.one_hot(torch.tensor(obs), num_classes=14)
        return state

    def save_model(self, name):
        torch.save(self.behavior_network.state_dict(), f'{"./NFSPAgent.pt"}')

    def update_behavior_policy(self, batch_data):
        batch_data_tensor = self._batch_to_tensor_OneHot(batch_data)
        obs_tensor = batch_data_tensor['obs'].float()
        actions_tensor = batch_data_tensor['action'].long()
        next_obs_tensor = batch_data_tensor['next_obs'].float()
        rewards_tensor = batch_data_tensor['reward'].float()
        dones_tensor = batch_data_tensor['done'].float()

        modelQvalue = self.behavior_network(obs_tensor)
        Q_S_A_value = self.behavior_network(obs_tensor)
        Q_S_A_value = Q_S_A_value.gather(1, actions_tensor)
        q_behavior_idx = modelQvalue.max(1)[1]
        q_behavior_idx = q_behavior_idx.unsqueeze(1)

        # compute the TD target using the target network
        q_values_next_state = self.target_network(next_obs_tensor).detach()
        q_value_max_next_S = q_values_next_state.gather(1, q_behavior_idx)

        undones_tensor = 1 - dones_tensor
        td = rewards_tensor + self.gamma * q_value_max_next_S * undones_tensor

        # compute the loss
        loss = nn.MSELoss()
        td_loss = loss(Q_S_A_value, td)

        # minimize the loss
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss.item()

    def update_target_policy(self):
        self.target_network = self.behavior_network

    def update_policy(self, batch_data):
        batch_data_tensor = self._batch_to_tensor_OneHot(batch_data)
        obs_tensor = batch_data_tensor['obs'].float()
        actions_tensor = batch_data_tensor['action'].long()
        actions = self.policy(obs_tensor)
        idx = actions_tensor
        actions = actions.gather(1, idx)
        log_val = actions.log()
        policy_loss = -1 * (log_val.mean())
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss

    def _batch_to_tensor_OneHot(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        # get the numpy arrays
        obs_arr, action_arr, reward_arr, next_obs_arr, done_arr = batch_data
        # convert to tensors
        for obs in obs_arr:
            batch_data_tensor['obs'].append(self.one_hot_encoding(obs, 14))
        batch_data_tensor['obs'] = torch.as_tensor((batch_data_tensor['obs']))

        #for action in action_arr:
        #    batch_data_tensor['action'].append(self.one_hot_encoding([action], 15))
        #batch_data_tensor['action'] = torch.as_tensor((batch_data_tensor['action'])).long().view(-1, 1).to(self.device)

        batch_data_tensor['action'] = torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        #for r in reward_arr:
        #    batch_data_tensor['reward'].append(self.one_hot_encoding([r], 15))
        #batch_data_tensor['reward'] = torch.as_tensor((batch_data_tensor['reward'])).long().view(-1, 1).to(self.device)

        for nx_obs in obs_arr:
            batch_data_tensor['next_obs'].append(self.one_hot_encoding(nx_obs, 14))
        batch_data_tensor['next_obs'] = torch.as_tensor((batch_data_tensor['next_obs']))

        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        return batch_data_tensor