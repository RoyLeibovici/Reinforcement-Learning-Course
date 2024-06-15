"""
    Based on: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
    The goal of the CartPole environment is to balance the pole on the cart by moving the cart left or right.
    The environment is considered solved if the pole is balanced for 200 time steps.
    The environment is considered failed if the pole falls down or the cart moves out of bounds.
    The CartPole environment is considered solved if the average reward is greater than or equal to 195.0 over 100 consecutive episodes.
    The average reward is calculated as the sum of rewards in 100 consecutive episodes divided by 100.

    The code will open a window and you will see the cartpole environment in action.
    The cartpole environment will be rendered for 200 time steps.
    The agent will choose an action based on the custom agent's policy.
    The agent will take the chosen action and the environment will return the next state, reward, and done flag.

    The CartPole environment has the following 4 observations:
        1. Cart Position
        2. Cart Velocity
        3. Pole Angle
        4. Pole Angular Velocity

    The CartPole environment has 2 actions:
        1. Move the cart to the left - Action 0
        2. Move the cart to the right - Action 1
        
"""

import gym
import numpy as np


class CustomAgent:
    def __init__(self, observation_space):
        self.observation_space = observation_space
        self.weights = np.random.uniform(-1, 1, size=self.observation_space.shape)

    def get_action(self, observation):
        observation_weight_product = np.dot(observation, self.weights)
        return 1 if observation_weight_product >= 0 else 0


# load CartPole's environment
env = gym.make('CartPole-v1', render_mode="human")
env.action_space.seed(42)

# Create the custom agent
agent = CustomAgent(env.observation_space)

# reset the environment
observation, info = env.reset(seed=42)

# calculate reward of the episode
total_reward = 0

# run the environment
for i in range(200):
    print("Episode Step:", i)
    
    # Render the environment to visualize it
    env.render()

    # Choose action based on the custom agent's policy
    action = agent.get_action(observation)
    
    # Take the chosen action
    next_state, reward, done, info, debug_info = env.step(action)

    print("Chosen action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)

    # Update the current observation
    observation = next_state

    # Update the total reward
    total_reward += reward

    # Check if the episode is done
    if done:
        print("Total Reward:", total_reward)
        print("Done:", done)
        break
        
env.close()
