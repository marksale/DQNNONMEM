import tensorflow as tf
print(" we are currently using Tensorflow version {}".format(tf.__version__))
import numpy as np
import random
#from IPython.display import clear_output
from collections import deque
import MyFrozenLakeEnv as MyEnv
#import progressbar

#import gym

#from tensorflow.keras import Model, Sequential
#from tensorflow.keras.layers import Dense, Reshape
#from tensorflow.keras.optimizers import Adam, SGD
#from tensorflow import keras


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
#from tensorflow import keras


#env = gym.make("FrozenLake8x8-v0")
#env = gym.make("FrozenLake-v0")
env = MyEnv.FrozenLakeEnv()
env.render()
print('Number of states: {}'.format(env.observation_space.n))
print("number of actions: {}".format(env.action_space.n))

# make some changes for git
class Q_learn:
  def __init__(self, env, optimizer, episodes, explore):
    self._state_size = env.observation_space.n
    self._action_size = env.action_space.n
    self._optimizer = optimizer
    self.gamma = 0.9
    self.episodes = episodes
    self.epsilon = explore
    #We save the experience for success and failures induvidually. 
    #Usually there are two many faliure and we get rewards only when we readch the goal. There are no intermediate rewards.
    # Consequently, distribution over success /failures are imbalanced.
    #So training the algo on failures is useful, only if we have some date where we succeeded.
    self.experience_replay_s = deque(maxlen = 2000) 
    self.experience_replay_f = deque(maxlen = 2000)

    self.q_network = self.network_model()
    self.target_network = self.network_model()
    self.copy_weights()

  def copy_weights(self):
    """
    copies weigths from current Q network to a target Q network
    """
    self.target_network.set_weights(self.q_network.get_weights())

  def store(self, state, action, reward, next_state, terminated):
    """
    We save the experience for success and failures induvidually. 
    """
    if reward>0.0:
      self.experience_replay_s.append((state, action, reward, next_state, terminated))
    else:
      self.experience_replay_f.append((state, action, reward, next_state, terminated))

  def eps_policy(self, state):
    """
    Episilon greedy policy
    """
    if np.random.rand() <= self.epsilon:
      return env.action_space.sample()
    else:
      q_values = self.q_network.predict(np.array([state]))
      return np.argmax(q_values[0])

  def network_model(self):
    """
    NN model for Q value function
    """
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=[1]))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(self._action_size, activation='relu'))
    model.compile(loss = 'mse', optimizer = self._optimizer)
    return model
  

  def test_fun(self, state, action, reward, next_state, terminated):
    """
    using s, a, r, s' we compute the target
    """
    target = self.q_network.predict(np.array([state]))
    if terminated:
      target[0][int(action)] = reward
    else:
      #A = np.ones(self._action_size, dtype=float)*self.epsilon/self._action_size # [eps/n ....n.... eps/n] 
      #q_values = self.target_network.predict(np.array([state]))
      #best_action = np.argmax(q_values[0])
      #A[best_action] += (1.0 - self.epsilon)
      #A captures the prop distribution of actions for a given state
      q_val_next_state = self.target_network.predict(np.array([next_state]))
      # Now take the Expectation of Next Q value over all possible actions of the given policy A
      #Expected_next_q_value = np.sum(np.array(q_val_next_state)*A)
      target[0][int(action)] = reward + self.gamma*np.amax(q_val_next_state)
    return list(target[0])

  def train(self, batch_size):
    """
    we randomly sample a batch with replacement from both successful and and failure buffers
    We fit the Q network with this batch
    """
    minibatch_s = np.array(random.choices(self.experience_replay_s, k = batch_size))
    minibatch_f = np.array(random.choices(self.experience_replay_f, k = batch_size))
    minibatch = np.concatenate((minibatch_s,minibatch_f), axis =0)
    batch_state = minibatch[:,0]
    batch_target = np.array([self.test_fun(state, action, reward, next_state, terminated) for state, action, reward, next_state, terminated in minibatch])
      
    self.q_network.fit(batch_state, batch_target, epochs = 1, verbose = 1)
    
optimizer = Adam(learning_rate=0.01)
test_agent = Q_learn(env, optimizer, episodes= 100, explore =0.9)
 
print("Testing the model:")
test_agent.q_network.build()
print(test_agent.q_network.summary())
test_predicted = test_agent.q_network.predict(np.array([env.observation_space.sample()]))
print("predict the q_values for a random sample is {}".format(test_predicted))
test_history = test_agent.q_network.fit(np.array([env.observation_space.sample()]), test_predicted, epochs=1)

optimizer = Adam(learning_rate=0.01)
#optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

agent = Q_learn(env, optimizer,episodes=1000000, explore = 0.9)

batch_size = 2000
num_of_episodes = 1000000
timesteps_per_episode = 128
Fail = 0
S = 0
for epi in range(0,num_of_episodes):
  state = env.reset()

  reward = 0
  if S>=1000:
    agent.epsilon = 0.1
  else:
    #until we get 50 success we explore a lot
    agent.epsilon = 0.99

  terminated  = False


  for steps in range(timesteps_per_episode):
    action = agent.eps_policy(state)
    next_state, reward, terminated, _  = env.step(action)
    agent.store(state, action, reward, next_state, terminated)
    state = next_state

    
    #bar.finish()
    if S>=1000:
        print("**********************************")
        env.render()
        print("**********************************")
    if terminated:
      if reward > 0.0:
        S=S+1
      else:
        Fail = Fail +1
      break
  if S>=1000:
    """
    we train only after getting enough experience
    """
    agent.train(batch_size)
    agent.copy_weights()
    print("Episode: {}, Success = {}, Failures = {}, exploration ={}".format(epi + 1, S, Fail, agent.epsilon))
  if epi%100 ==0:
    print("Episode: {}, Success = {}, Failures = {}, exploration ={}".format(epi + 1, S, Fail, agent.epsilon))
    


