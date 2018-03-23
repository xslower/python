from header import *
from sklearn import preprocessing


from stock_simulator import Simulator
from rl_dqn import Dqn

stock_list = [1, 2, 4, 5, 7, 8, 9]
obs_len = 300
epoch_num = 200
act_num = 2


def train_test_split(dx, rate = 9):
    x = np.array(dx)
    pos = len(x) // 10 * rate
    return x[:pos], x[pos:]


def run(train_end):
    for epoch in range(epoch_num):
        sim.reset()
        last_store = 0
        log.info('epoch %s', epoch)
        if epoch < 10:
            dqn.rand_gate = 0.0
        else:
            if epoch % 2 == 0:
                dqn.rand_gate = 0.5
            else:
                dqn.rand_gate = 1
        for idx in range(train_end - 1):
            if samples[idx] is None:
                continue
            act = dqn.train_action(idx, last_store)
            store, reward = sim.step(idx, act)
            dqn.store_transition(idx, last_store, act, reward, next_store=store)
            last_store = store
            if idx % 10 == 0:
                cost = dqn.learn()
                log.info('cost: %s', cost)
            # if epoch > 2:
            sim.show_act(idx, act)


def test(start, end):
    log.info('test epoch! start %s end %s!!!!!!!!!!!!!!!!!!!', start, end)
    store = 0
    sim.reset()
    dqn.rand_gate = 1
    for idx in range(start, end):
        obs = samples[idx]
        act = dqn.pred_action(obs, store)
        store, reward = sim.step(idx, act)
        sim.show_act(idx, act)


if __name__ == '__main__':
    scaler = preprocessing.StandardScaler()
    k_line, samples = stock_data.prepare_single(1)
    sim = Simulator(k_line)
    rate = 5
    split = len(samples) // 10 * rate
    dqn = Dqn(act_num, samples, k_line)
    # strain, stest = train_test_split(samples, 6)
    run(split)
    test(0, split)
    test(split, len(samples))
