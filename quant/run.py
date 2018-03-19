import numpy as np
from sklearn import preprocessing

import stock_data
from stock_simulator import Simulator
from rl_dqn import Dqn

stock_list = [1, 2, 4, 5, 7, 8, 9]
obs_len = 300
epoch_num = 100

def train_test_split(dx, rate = 9):
    x = np.array(dx)
    pos = len(x) // 10 * rate
    return x[:pos], x[pos - obs_len:]


def run(strain):
    sim = Simulator(strain)
    for epoch in range(epoch_num):
        sim.reset()
        for idx in range(obs_len, len(strain) - 1):
            obs = strain[idx - obs_len:idx + 1]
            obs = scaler.fit_transform(obs)
            obs_now = obs[:obs_len]
            act = dqn.choose_action(obs_now)
            reward = sim.step(idx, act)
            obs_nxt = obs[1:]
            dqn.store_transition(obs_now, act, reward, next_obs=obs_nxt)
            if idx % 5 == 0:
                dqn.learn()
            sim.show_act(idx, act)


def test(stest):
    sim = Simulator(stest)
    for idx in range(obs_len, len(stest)):
        obs = stest[idx - obs_len:idx]
        obs = scaler.fit_transform(obs)
        act = dqn.choose_action(obs)
        sim.step(idx, act)
        sim.show_act(idx, act)


if __name__ == '__main__':
    scaler = preprocessing.StandardScaler()
    sdata = stock_data.load_file(1)
    strain, stest = train_test_split(sdata, 7)
    dqn = Dqn(3, [300, 6])
    run(strain)
    test(stest)
