import random
import numpy as np
import matplotlib.pyplot as plt

class StockPriceGenerator(object):
    def __init__(self, init_pr_spot, interest, vol):
        self.init_pr_spot = init_pr_spot # initial spot price
        self.interest = interest # risk-free interest rate
        self.vol = vol # volatility

    def closed_form(self, n_data, n_step, time_horizon = 1):
        dt = time_horizon / n_step
        bm_step = np.random.normal(scale = np.sqrt(dt), size = n_data * n_step)
        bm_step = bm_step.reshape((n_data, n_step))
        W = np.cumsum(bm_step, 1) # Brownian motions: (n_data, n_step)
        t = np.arange(0, time_horizon, dt)

        tmp = (self.interest - 0.5 * np.power(self.vol, 2)) * t + self.vol * W
        pr_final = self.init_pr_spot * np.exp(tmp)
        return pr_final

    def euler_maruyama(self, n_path, n_step, time_horizon = 1):
        sampled_paths = None
        dt = time_horizon / n_step
        sampled_paths = np.zeros((n_path, n_step))
        sampled_paths[:, 0] = self.init_pr_spot

        for i in range(1, n_step):
            w = np.random.standard_normal(n_path)
            offset = self.interest * dt + vol * np.sqrt(dt) * w
            sampled_paths[:, i] = sampled_paths[:, i - 1] * (1 + offset)

        return sampled_paths

class AsianOptionPricing(object):
    def __init__(self):
        pass

    def option_eur(self, pr_strike):
        pass

    @staticmethod
    def discrete(sampled_paths, pr_strike, interest, day_expire, arithmetic_avg, fix_strike):
        if arithmetic_avg:
            avg = np.mean(sampled_paths, 1) # shape = (#path, )
        else:
            avg = np.exp(np.mean(np.log(sampled_paths), 1)) # shape = (#path, )

        if fix_strike:
            all_pr_call = np.maximum(0, avg - pr_strike)
            all_pr_put = np.maximum(0, pr_strike - avg)
        else:
            pr_stock_final = sampled_paths[:, -1]
            all_pr_call = np.maximum(0, pr_stock_final - avg)
            all_pr_put = np.maximum(0, avg - pr_stock_final)

        decay = np.exp(-interest * day_expire)
        pr_call = decay * np.mean(all_pr_call)
        pr_put = decay * np.mean(all_pr_put)

        return pr_call, pr_put


if __name__ == "__main__":
    seed = 2023
    np.random.seed(seed)
    random.seed(seed)

    init_pr_spot = 100
    pr_strike = 100
    interest = 0.05
    vol = 0.2
    horizon = 1
    day_expire = 252
    n_path = int(1e5)
    generator = StockPriceGenerator(init_pr_spot, interest, vol)
    price_paths = generator.euler_maruyama(n_path, day_expire)
    closed_form = generator.closed_form(n_path, day_expire)

    arithmetic_avg = True
    fix_strike = True
    pr_call, pr_put = AsianOptionPricing.discrete(price_paths, pr_strike, interest, day_expire, arithmetic_avg, fix_strike)
    print(pr_call, pr_put)
