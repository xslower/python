'''股市环境模型'''

from header import *


class Simulator(object):
    def __init__(self, env, cash = 100000):
        self.total_cash = cash
        self.action_space = ['sell', 'buy', 'wait']
        self.num_acts = len(self.action_space)
        self.stock_val = 0
        # 配置股票数据
        self.env = env
        self.reset()

    def _clse_up(self, idx):
        up = self.env[idx][stock_data.O_CLSE_UP]
        return up
    def _open_up(self, idx):
        return self.env[idx][stock_data.O_OPEN_UP]

    def reset(self):
        self.cash = self.total_cash
        self.cost = self.cash
        # self.day_idx = 0
        self.stock_val = 0
        self.last_act = -1

    def reward(self, idx, act):
        fee = 0.99
        up = self._clse_up(idx)
        oup = self._open_up(idx)
        stock = self.stock_val
        cash = self.cash
        reward = 0
        if act == 0:  # 卖出
            if stock > 0:
                stock *= (1 + oup)
                cash += stock * fee
                stock = 0
                mini = cash if cash < self.cost else self.cost
                reward = (cash - self.cost) / mini * 10
            else:  # do nothing
                pass
        else:  # 买入
            # 持股价值更新
            stock *= (1 + up)
            if cash > 0:
                self.cost = cash
                tmp_stock = cash * fee
                # if up < 0:
                tmp_stock *= (1 + up - oup)
                stock += tmp_stock
                cash = 0
            else:  # do nothing
                pass
        self.stock_val = stock
        self.cash = cash
        return reward

    def step(self, idx, act):
        store = act
        reward = self.reward(idx, act)
        return store, reward

    def act_name(self, act):
        if act > 2:
            print(act)
        return self.action_space[act]

    def show_act(self, idx, act):
        an = self.act_name(act)
        if act != self.last_act:
            log.info('%s %s stock:%.2f cash:%.2f', idx, an, self.stock_val, self.cash)
        self.last_act = act
