import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(DQN, self).__init__()
        self.inputLayer = nn.Linear(input_size, hidden_layer_size)
        self.relu_fn_1 = nn.ReLU()
        self.hiddenLayer1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.relu_fn_2 = nn.ReLU()
        self.hiddenLayer2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.relu_fn_3 = nn.ReLU()
        self.outputLayer = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.relu_fn_1(x)
        x = self.hiddenLayer1(x)
        x = self.relu_fn_2(x)
        x = self.hiddenLayer2(x)
        x = self.relu_fn_3(x)
        y = self.outputLayer(x)
        return y


class DQNAgent(object):
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
        self.gamma = 0.99
        self.optimizer = torch.optim.Adam(self.behavior_network.parameters(), lr=parameters['learning_rate'])
        self.device = "cpu"

    def get_action(self, obs, eps, hand):
        eps = 0.1
        if np.random.random() < eps:
            action = np.random.choice(hand, 1)[0]
            return action.run_val
        else:
            obs = self.toTensorOneHot(obs).view(1, -1)
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
                # action = q_values.max(dim=1)[1].item()
            if action == 0:
                action = 1
            if action == 14:
                action = 13
            return action

    def update_behavior_policy(self, batch_data):
        # batch_data_to one_hot
        batch_tensor = self._batch_to_tensor_OneHot(batch_data)
        obs_tensor = batch_tensor['obs'].float()
        actions_tensor = batch_tensor['action'].long()
        next_obs_tensor = batch_tensor['next_obs'].float()
        rewards_tensor = batch_tensor['reward'].float()
        dones_tensor = batch_tensor['done'].float()

        modelQvalue = self.behavior_network(obs_tensor).gather(1, actions_tensor)
        q_values_next_state = self.target_network(next_obs_tensor).detach()
        q_value_max_next_S = q_values_next_state.max(1)[0]

        q_value_max_next_S_dim = q_value_max_next_S.unsqueeze(1)
        undones_tensor = 1 - dones_tensor
        td = rewards_tensor + self.gamma * q_value_max_next_S_dim * undones_tensor

        loss = nn.MSELoss()
        td_loss = loss(modelQvalue, td)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
        return td_loss.item()

    def update_target_policy(self):
        self.target_network = self.behavior_network

    def toTensorOneHot(self, obs):
        state = F.one_hot(torch.tensor(obs), num_classes=14)
        return state

    def _batch_to_tensor_OneHot(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        # get the numpy arrays
        obs_arr, action_arr, reward_arr, next_obs_arr, done_arr = batch_data
        # convert to tensors
        for obs in obs_arr:
            batch_data_tensor['obs'].append(self.one_hot_encoding(obs, 14))
        batch_data_tensor['obs'] = torch.as_tensor((batch_data_tensor['obs']))
        # for action in action_arr:
        #    batch_data_tensor['action'].append(self.one_hot_encoding([action], 15))
        # batch_data_tensor['action'] = torch.as_tensor((batch_data_tensor['action'])).long().view(-1, 1).to(self.device)

        batch_data_tensor['action'] = torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        # for r in reward_arr:
        #    batch_data_tensor['reward'].append(self.one_hot_encoding([r], 15))
        # batch_data_tensor['reward'] = torch.as_tensor((batch_data_tensor['reward'])).long().view(-1, 1).to(self.device)

        for nx_obs in obs_arr:
            batch_data_tensor['next_obs'].append(self.one_hot_encoding(nx_obs, 14))
        batch_data_tensor['next_obs'] = torch.as_tensor((batch_data_tensor['next_obs']))

        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        return batch_data_tensor

    # One hot encoding
    def one_hot_encoding(self, obs, num):
        y = np.array([])
        for i in obs:
            x = np.zeros(num)
            x[i] = 1
            y = np.concatenate((y, x), axis=None)
        return y

    # Save model
    def save_model(self):
        torch.save(self.behavior_network.state_dict(), f'{"./DQN_data.pt"}')
