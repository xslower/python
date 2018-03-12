"""
Double DQN & Natural DQN comparison,
The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from maze_env import Maze

# env = gym.make('Pendulum-v0')
# env = env.unwrapped
# env.seed(1)
MEMORY_SIZE = 3000
env = Maze()
ACTION_SPACE = env.n_actions
N_FEATURES = env.n_features

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=N_FEATURES, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=N_FEATURES, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    step = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()
        env.render()
        action = RL.choose_action(observation)
        observation_, reward, done = env.step(action)

        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        observation = observation_

        if done:
            break
        step += 1

    return RL.q

if __name__ == "__main__":
    q_natural = train(natural_DQN)
    # q_double = train(double_DQN)
    env.mainloop()
    plt.plot(np.array(q_natural), c='r', label='natural')
    # plt.plot(np.array(q_double), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
