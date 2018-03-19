'''股市环境模型'''

class Simulator(object):
    def __init__(self, env, cash = 100000):
        self.total_cash = cash
        self.action_space = ['buy', 'sell', 'wait']
        self.num_acts = len(self.action_space)
        self.stock_val = 0
        # 配置股票数据
        self.env = env
        self.reset()

    def _up(self, idx):
        return self.env[idx][1]

    def reset(self):
        self.cash = self.total_cash
        self.day_idx = 0
        self.stock_val = 0
        self.last_act = -1

    def step(self, idx, act):
        fee = 0.995
        min_val = 10
        before = self.stock_val + self.cash
        up = self._up(idx)
        action = self.action_space[act]
        if action == 'buy':  # buy
            if self.cash > min_val:
                # 买入卖出都扣除0.5％的手续费
                # 以当天开盘价和收盘价中的最高价买入
                if up >= 0:
                    self.stock_val += self.cash * fee
                else:  # 相当于当天买入就亏损
                    self.stock_val += self.cash * (fee + up)
                self.cash = 0
            else:  # 没钱买就相当于持仓观望
                self.stock_val += self.stock_val * up
        elif action == 'sell':  # sell
            if self.stock_val > min_val:
                # 以当天起止两点最低价卖出
                if up >= 0:
                    self.cash += self.stock_val * fee
                else:
                    self.cash += self.stock_val * (fee + up)
                self.stock_val = 0
            else:  # 没股票卖相当于持币观望
                self.stock_val += self.stock_val * up
        else:  # wait
            self.stock_val += self.stock_val * up
        reward = (self.stock_val + self.cash) / before - 1
        return reward

    def act_name(self, act):
        if act > 3:
            print(act)
        return self.action_space[act]

    def show_act(self, idx, act):
        an = self.act_name(act)
        if an != 'wait' and act != self.last_act:
            print(idx, an, self.stock_val, self.cash)
        self.last_act = act