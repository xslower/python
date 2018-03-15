class Simulator(object):
    def __init__(self, env, cash = 100000):
        self.total_cash = cash
        self.action_space = ['buy', 'sell', 'wait']
        self.num_acts = len(self.action_space)
        self.stock = 0
        # 配置股票数据
        self.env = env
        self.reset()

    def reset(self):
        self.cash = self.total_cash
        self.day_idx = 0
        self.stock = 0

    def reward(self, idx, action):
        up = self.env[idx]


    def step(self, idx, action):
        self.day_idx += 1
        if action == 0:  # buy
            if self.cash > 10:
