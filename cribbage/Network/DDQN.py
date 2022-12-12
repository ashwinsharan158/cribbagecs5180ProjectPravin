from .DQNAgent import DQN
import numpy as np
import torch
from torch import nn


# Double DQN Agent
class DDQNAgent(object):
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

        self.optimizer = torch.optim.Adam(self.behavior_network.parameters(), lr=parameters['learning_rate'])
        self.device = "cpu"
        self.gamma = 0.99

    def get_action(self, obs, eps, hand):
        if np.random.random() < eps:
            action = np.random.choice(hand, 1)[0]
            return action.run_val
        else:
            obs = torch.from_numpy(np.array(obs)).float().to(self.device).view(1, -1)
            with torch.no_grad():
                q_values = self.behavior_network(obs)
                action = q_values.max(dim=1)[1].item()
            return action + 1

    def update_behavior_policy(self, batch_data):
        batch_tensor = self.get_batch(batch_data)
        obs_tensor = batch_tensor['obs']
        actions_tensor = batch_tensor['action']
        next_obs_tensor = batch_tensor['next_obs']
        rewards_tensor = batch_tensor['reward']
        dones_tensor = batch_tensor['done']

        # Behavior S,A selection
        modelQvalue = self.behavior_network(obs_tensor)
        Q_S_A_value = self.behavior_network(obs_tensor).gather(1, actions_tensor)
        q_behavior_idx = modelQvalue.max(1)[1]
        q_behavior_idx = q_behavior_idx.unsqueeze(1)

        # Target
        q_values_next_state = self.target_network(next_obs_tensor).detach()
        q_value_max_next_S = q_values_next_state.gather(1, q_behavior_idx)
        undones_tensor = 1 - dones_tensor
        td = rewards_tensor + self.gamma * q_value_max_next_S * undones_tensor

        # loss
        loss = nn.MSELoss()
        td_loss = loss(Q_S_A_value, td)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss.item()

    def update_target_policy(self):
        self.target_network = self.behavior_network

    # Batch data to tensor
    def get_batch(self, data):
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


    def save_model(self):
        torch.save(self.behavior_policy_net.state_dict(), f'{"./DDQN_data.pt"}')
