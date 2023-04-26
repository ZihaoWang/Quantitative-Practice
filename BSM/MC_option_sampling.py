import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class StockPriceGenerator(object):
    def __init__(self, init_pr_spot, interest, vol):
        self.init_pr_spot = init_pr_spot # initial spot price
        self.interest = interest # risk-free interest rate
        self.vol = vol # volatility

    def closed_form(self, n_path, n_step, time_horizon = 1, dW = None):
        dt = time_horizon / n_step
        if dW is None:
            dW = np.sqrt(dt) * np.random.standard_normal(n_path * n_step)
            dW = dW.reshape((n_path, n_step))
        W = np.cumsum(dW, 1) # Brownian motions: (n_path, n_step)
        t = np.arange(0, time_horizon, dt)

        tmp = (self.interest - 0.5 * np.power(self.vol, 2)) * t + self.vol * W
        sampled_paths = self.init_pr_spot * np.exp(tmp)
        sampled_paths = np.concatenate([np.ones((n_path, 1)) * self.init_pr_spot, sampled_paths], 1)

        if n_path == 1:
            sampled_paths = sampled_paths.squeeze(0)
        return sampled_paths

    def euler_maruyama(self, n_path, n_step, time_horizon = 1, dW = None):
        dt = time_horizon / n_step
        if dW is None:
            dW = np.sqrt(dt) * np.random.standard_normal(n_path * n_step)
            dW = dW.reshape((n_path, n_step))
        sampled_paths = np.zeros((n_path, n_step + 1))
        sampled_paths[:, 0] = self.init_pr_spot

        for i in range(1, n_step + 1):
            offset = self.interest * dt + vol * dW[:, i - 1]
            sampled_paths[:, i] = sampled_paths[:, i - 1] * (1 + offset)

        if n_path == 1:
            sampled_paths = sampled_paths.squeeze(0)
        return sampled_paths

    def milstein(self, n_path, n_step, time_horizon = 1, dW = None):
        dt = time_horizon / n_step
        if dW is None:
            dW = np.sqrt(dt) * np.random.standard_normal(n_path * n_step)
            dW = dW.reshape((n_path, n_step))
        sampled_paths = np.zeros((n_path, n_step + 1))
        sampled_paths[:, 0] = self.init_pr_spot

        for i in range(1, n_step + 1):
            offset = self.interest * dt + vol * dW[:, i - 1] + 0.5 * np.power(vol, 2) * (np.power(dW[:, i - 1], 2) - dt)
            sampled_paths[:, i] = sampled_paths[:, i - 1] * (1 + offset)

        if n_path == 1:
            sampled_paths = sampled_paths.squeeze(0)
        return sampled_paths

    def plot_convergence(self):
        strong_err_eu, strong_err_mil, weak_err_eu, weak_err_mil = [], [], [], []
        all_step = np.array([10, 20, 40, 100, 200, 400, 1000])
        n_path = 100

        for n_step in all_step:
            err_eu, err_mil = np.zeros(n_step + 1), np.zeros(n_step + 1)
            exacted_sum, eu_sum, mil_sum = np.zeros(n_step + 1), np.zeros(n_step + 1), np.zeros(n_step + 1)

            for i_data in range(n_path):
                # use the same BM for three methods, otherwise too noisy
                np.random.seed(i_data)
                dt = 1 / n_step
                dW = np.sqrt(dt) * np.random.standard_normal(n_step)
                dW = dW.reshape((1, n_step))

                exacted_path = self.closed_form(1, n_step, dW = dW)
                eu_path = self.euler_maruyama(1, n_step, dW = dW)
                mil_path = self.milstein(1, n_step, dW = dW)

                err_eu += np.abs(eu_path - exacted_path)
                err_mil += np.abs(mil_path - exacted_path)
                exacted_sum += exacted_path
                eu_sum += eu_path
                mil_sum += mil_path

            strong_err_eu.append(np.max(err_eu / n_path))
            strong_err_mil.append(np.max(err_mil / n_path))
            weak_err_eu.append(np.max(np.abs((exacted_sum - eu_sum) / n_path)))
            weak_err_mil.append(np.max(np.abs((exacted_sum - mil_sum) / n_path)))

        figure = plt.figure(num = 1, figsize = (14, 8))
        ax = figure.add_subplot(111)
        dt = 1 / all_step
        ax.loglog(dt, strong_err_eu, label = "EM error: strong", color = "red")
        ax.loglog(dt, weak_err_eu, label = "EM error: weak", color = "orange", ls = "--")
        ax.loglog(dt, strong_err_mil, label = "Milstein error: strong", color = "blue")
        ax.loglog(dt, weak_err_mil, label = "Milstein error: weak", color = "green", ls = "--")
        ax.set_xlabel("Time inverval $\Delta t$")
        ax.set_ylabel("Error")
        ax.legend()
        figure.savefig("/Users/evensong/Desktop/CQF/exams/exam2/fig/convergency_error.png", format = "png")
        plt.show()



class AsianOptionPricing(object):
    def __init__(self, sampled_paths, interest, day_expire):
        self.sampled_paths = sampled_paths
        self.interest = interest
        self.day_expire = day_expire

    def discrete(self, pr_strike, arithmetic_avg, fix_strike):
        if arithmetic_avg:
            avg = np.mean(self.sampled_paths, 1) # shape = (#path, )
        else:
            avg = np.exp(np.mean(np.log(self.sampled_paths), 1)) # shape = (#path, )

        if fix_strike:
            all_pr_call = np.maximum(0, avg - pr_strike)
            all_pr_put = np.maximum(0, pr_strike - avg)
        else:
            pr_stock_final = self.sampled_paths[:, -1]
            all_pr_call = np.maximum(0, pr_stock_final - avg)
            all_pr_put = np.maximum(0, avg - pr_stock_final)

        decay = np.exp(-self.interest) # 1-year expiration corresponds to 1-year interest rate
        pr_call = decay * np.mean(all_pr_call)
        pr_put = decay * np.mean(all_pr_put)

        return pr_call, pr_put

    def plot_prices(self, pr_strikes):
        i_figure = 1
        fixed_strike = True
        for arithmetic_avg in [True, False]:
            pr_calls, pr_puts = [], []
            for pr_strike in pr_strikes:
                pr_call, pr_put = self.discrete(pr_strike, arithmetic_avg, fixed_strike)
                pr_calls.append(pr_call)
                pr_puts.append(pr_put)

            figure = plt.figure(num = i_figure, figsize = (14, 8))
            ax = figure.add_subplot(111)
            ax.plot(pr_strikes, pr_calls, label = "Asian call prices", color = "blue")
            ax.plot(pr_strikes, pr_puts, label = "Asian put prices", color = "orange")
            ax.set_xlabel("Strike price")
            ax.set_ylabel("Option price")
            ax.legend()
            fname = "asian_option_"
            fname += "arithmetic_" if arithmetic_avg else "geometric_"
            fname += "fixed_strike"
            figure.savefig(f"/Users/evensong/Desktop/CQF/exams/exam2/fig/{fname}.png", format = "png")
            i_figure += 1

        fixed_strike = False
        prices = []
        avg_labels = []
        for arithmetic_avg in [True, False]:
            pr_call, pr_put = self.discrete(pr_strike, arithmetic_avg, fixed_strike)
            avg_label = "arithmetic_avg" if arithmetic_avg else "geometric_avg"
            prices += [pr_call, pr_put]
            avg_labels += [f"call ({avg_label})", f"put ({avg_label})"]

        figure = plt.figure(num = i_figure, figsize = (14, 8))
        ax = figure.add_subplot(111)
        ax.scatter(avg_labels, prices, color = ["orange", "red", "green", "blue"])
        ax.set_xlabel("Floating strike Asian options")
        ax.set_ylabel("Option price")
        fname = "asian_option_float_strike"
        figure.savefig(f"/Users/evensong/Desktop/CQF/exams/exam2/fig/{fname}.png", format = "png")




if __name__ == "__main__":
    np.set_printoptions(precision = 2)
    seed = 1024
    np.random.seed(seed)
    random.seed(seed)

    init_pr_spot = 100
    interest = 0.05
    vol = 0.2
    generator = StockPriceGenerator(init_pr_spot, interest, vol)
    #generator.plot_convergence()

    day_expire = 252
    n_path = int(1e4)
    pr_strikes = np.arange(10, 201, 10)
    milstein_paths = generator.euler_maruyama(n_path, day_expire)
    asian = AsianOptionPricing(milstein_paths, interest, day_expire)
    asian.plot_prices(pr_strikes)
