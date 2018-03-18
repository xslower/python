class Simulator(object):
    def __init__(self, env, cash = 100000):
        self.total_cash = cash
        self.action_space = ['buy', 'sell', 'wait']
        self.num_acts = len(self.action_space)
        self.stock_val = 0
        # 配置股票数据
        self.env = env
        self.reset()

    def reset(self):
        self.cash = self.total_cash
        self.day_idx = 0
        self.stock_val = 0

    def step(self, idx, action):
        fee = 0.995
        min_val = 1000
        before = self.stock_val + self.cash
        up = self.env[idx]
        # self.day_idx += 1
        if action == 0:  # buy
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
        elif action == 1:  # sell
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
        return self.stock_val, reward
