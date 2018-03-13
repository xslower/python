from maze_env import Maze
from RL_brain import DuelingDQN
import tensorflow as tf

def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    # RL = DeepQNetwork(env.n_actions, env.n_features,
    #                   memory_size=2000,
    #                   learning_rate=0.01,
    #                   reward_decay=0.9,
    #                   e_greedy=0.9,
    #                   replace_target_iter=200,
    #                   # output_graph=True
    #                   )
    # ddqn
    # RL = DoubleDQN(n_actions=env.n_actions, n_features=env.n_features, memory_size=2000, learning_rate=0.01, reward_decay=0.9,e_greedy_increment=0.001, e_greedy=0.9, double_q=True)
    # dqnpr
    # RL = DQNPrioritizedReplay(n_actions=env.n_actions, n_features=env.n_features, memory_size=2000, learning_rate=0.01, reward_decay=0.9,e_greedy_increment=0.001, e_greedy=0.9, prioritized=True)
    RL = DuelingDQN(n_actions=env.n_actions, n_features=env.n_features, memory_size=2000, learning_rate=0.01, reward_decay=0.9, e_greedy_increment=0.001, e_greedy=0.9, dueling=True)
    # sess.run(tf.global_variables_initializer())
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()