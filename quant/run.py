from header import *
from sklearn import preprocessing

import stock_data
from stock_simulator import Simulator
from rl_dqn import Dqn

stock_list = [1, 2, 4, 5, 7, 8, 9]
obs_len = 300
epoch_num = 50
act_num = 2

def train_test_split(dx, rate = 9):
    x = np.array(dx)
    pos = len(x) // 10 * rate
    return x[:pos], x[pos:]


def run():
    for epoch in range(epoch_num):
        sim.reset()
        last_store = 0
        log.info('epoch %s', epoch)
        for idx in range(obs_len, split - 1):
            if samples[idx] is None:
                continue
            act = dqn.choose_action(idx, last_store)
            store, reward = sim.step(idx, act)
            dqn.store_transition(idx, last_store, act, reward, next_store=store)
            last_store = store
            if idx % 10 == 0:
                cost = dqn.learn()
                log.info('cost: %s', cost)
            # if epoch > 2:
            sim.show_act(idx, act)


def test():
    store = 0
    for idx in range(split, len(samples)):
        dqn.rand_gate = 1
        act = dqn.choose_action(idx, store)
        store, reward = sim.step(idx, act)
        sim.show_act(idx, act)


if __name__ == '__main__':
    scaler = preprocessing.StandardScaler()
    k_line, samples = stock_data.prepare_single(1)
    sim = Simulator(k_line)
    dqn = Dqn(act_num, samples, [obs_len, 6])
    rate = 6
    split = len(samples) // 10 * rate
    # strain, stest = train_test_split(samples, 6)
    run()
    test()
