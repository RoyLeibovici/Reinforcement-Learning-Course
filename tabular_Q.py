!pip install gymnasium
import gymnasium as gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v1')

# Implement Q-Table learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

# Set learning parameters
lr = 0.9
y = 0.08
epsilon = 0.99
epsilon_decay = 0.999
min_epsilon = 0.1
num_episodes = 2000

#create lists to contain total rewards and steps per episode
#jList = []
rList = []

for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    state, _  = s
    rAll = 0 # Total reward during current episode
    #d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
  
        # TODO: Implement Q-Learning
        #Choose an action by greedily (with noise) picking from Q table
        Q_max_action = np.argmax(Q[state])
        random_action = env.action_space.sample()
        action = Q_max_action if np.random.random(1) > epsilon else random_action

        #Get new state and reward from environment
        new_state, reward, done, truncated, info  = env.step(action)

        #Update Q-Table with new knowledge
        Q[state][action] += lr * (reward + y * (np.max(Q[new_state]) - Q[state][action]))
        state = new_state

        #Update total reward
        rAll += reward

        #Update episode if we reached the Goal State
        if state == 15:
          break
     
    if epsilon > min_epsilon:
      epsilon *= epsilon_decay

    rList.append(rAll)

print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print (Q)


# Evaluate the success rate
success = 0
for i in range(num_episodes):
    state, _ = env.reset()
    j = 0
    while j < 99:
        j += 1
        Q_max_action = np.argmax(Q[state])  # Follow the policy derived from the Q-table
        action = Q_max_action

        new_state, reward, done, truncated, info = env.step(action)
        state = new_state

        if state == 15:  # Goal state
            success += 1
            break
        if done:
            break

print("Percent of successful episodes: " + str(100 * success / num_episodes) + '%')
