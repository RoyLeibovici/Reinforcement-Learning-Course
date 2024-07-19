#!pip install gymnasium
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

# Load environment
env = gym.make('FrozenLake-v1')

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss
# TODO: define network, loss and optimiser(use learning rate of 0.1).

num_actions = 4
num_states = 16
model = nn.Sequential(nn.Linear(num_states, num_actions, bias=False))
criterion = nn.MSELoss(reduce='None')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Implement Q-Network learning algorithm

# Set learning parameters
GAMMA = 0.99
epsilon = 0.9
epsilon_decay = 0.85
min_epsilon = 0.1
num_episodes = 2000

# create lists to contain total rewards and steps per episode
jList = []
rList = []
lossList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    state, _  = s
    rAll = 0
    done = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        # TODO: Implement Step 1
        state_one_hot = np.zeros(num_states)
        state_one_hot[state] = 1.0
        input = torch.from_numpy(state_one_hot)
        input = input.to(torch.float32)
        Q = model(input)
        action = np.argmax(Q.detach().numpy())

        # 2. A chance of e to perform random action
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()

        # 3. Get new state(mark as new_state) and reward(mark as reward) from environment
        new_state, reward, done, _, _ = env.step(action)

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        # TODO: Implement Step 4
        state_one_hot = np.zeros(num_states)
        state_one_hot[new_state] = 1.0
        input = torch.from_numpy(state_one_hot)
        input = input.to(torch.float32)
        Q_prime = model(input)

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        # TODO: Implement Step 5
        Q_target = np.zeros(num_actions)
        Q_target = torch.from_numpy(Q_target)
        Q_target = Q_target.to(torch.float32)
        action_prime = np.argmax(Q_prime.detach().numpy())

        Q_target = Q.clone().detach()
        Q_target[action] = reward + GAMMA * torch.max(Q_prime).item()

        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        # TODO: Implement Step 6
        optimizer.zero_grad()
        loss = criterion(Q_target, Q)
        lossList.append(loss.item())
        loss.backward()
        optimizer.step()

        rAll += reward
        state = new_state
        if done == True:
          break
    #Reduce chance of random action as we train the model.
    if epsilon > min_epsilon:
          epsilon *= epsilon_decay

    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))

# Plot the loss as a function of iteration
plt.plot(lossList)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.show()

# Evaluate the success rate
success = 0
for i in range(num_episodes):
    state, _ = env.reset()
    j = 0
    while j < 99:
        j += 1
        state_one_hot = np.zeros(num_states)
        state_one_hot[state] = 1.0
        input = torch.from_numpy(state_one_hot)
        input = input.to(torch.float32)
        Q = model(input)
        action = np.argmax(Q.detach().numpy())

        new_state, reward, done, truncated, info = env.step(action)
        state = new_state

        if state == 15:  # Goal state
            success += 1
            break
        if done:
            break

print("Percent of successful episodes: " + str(100 * success / num_episodes) + '%')
