import numpy as np
import matplotlib.pyplot as plt
import scipy


from scipy import stats
#from tabulate import tabulate

# Import plotly express
#import plotly.express as px
#px.defaults.width, px.defaults.height = 1000, 600

class VaR_Backtest(object):
    def __init__(self, n_backtest):
        self.date = []
        self.sp500 = []
        self.n_backtest = n_backtest
        #self.profit_rate = []

        self.read_data()
        self.sp500 = np.array(self.sp500)
        self.log_return = np.array([np.log(self.sp500[i]) - np.log(self.sp500[i - 1]) for i in range(1, len(self.sp500))])
        self.log_return = np.concatenate((np.array([np.nan]), self.log_return)) # align with self.date and self.sp500

        self.mu = np.mean(self.log_return[1:])
        self.sigma = np.std(self.log_return[1:])
        self.sigma_10day = self.sigma * np.sqrt(10)
        self.sigma_year = self.sigma * np.sqrt(252)
        print(f"mu = {self.mu}, 1 day sigma = {self.sigma}, 10 days sigma = {self.sigma_10day}, 1 year sigma = {self.sigma_year}")

    def comp_breach(self):
        self.__comp_var_10days()
        self.__comp_forward_return_10days()
        
        assert len(self.var_10days) == 10 + len(self.forward_return_10days)
        self.var_10days = self.var_10days[:-10]

        all_breaches = self.forward_return_10days < self.var_10days
        n_breach = np.sum(all_breaches)
        percent_breach = n_breach / all_breaches.shape[0]

        n_consecutive_breach = 0
        for i in range(0, all_breaches.shape[0] - 1):
            if np.abs(all_breaches[i] - 1.0) < 1e-6 and np.abs(all_breaches[i + 1] - 1.0) < 1e-6:
                n_consecutive_breach += 1
            percent_consecutive_breach = n_consecutive_breach / (all_breaches.shape[0] - 1)

        return n_breach, percent_breach, n_consecutive_breach, percent_consecutive_breach

    def plot_backtest(self):
        valid_date = self.date[n_backtest : n_backtest + len(self.forward_return_10days)]
        n_tick_date = 8
        tick_inverval = len(valid_date) // n_tick_date
        tick_dates = [valid_date[i * tick_inverval] for i in range(n_tick_date)]

        figure = plt.figure(num = 1, figsize = (10, 6))
        ax = figure.add_subplot(111)
        ax.plot(valid_date, self.var_10days, label = "10 day VaR")
        ax.plot(valid_date, self.forward_return_10days, label = "10 day realized")
        ax.set_title("VaR backtesting")
        ax.xaxis.set_ticks(np.arange(0, len(valid_date), tick_inverval))
        ax.legend()
        plt.show()

    def __comp_var_10days(self, percentile = 0.99):
        factor = -scipy.stats.norm.ppf(percentile) # inverse Gaussian CDF

        self.var_10days = []
        for i_begin in range(1, len(self.log_return)):
            i_end = i_begin + self.n_backtest
            if i_end >= len(self.log_return):
                break

            var = np.std(self.log_return[i_begin : i_end], ddof = 1) * np.sqrt(10) * factor
            self.var_10days.append(var)
            '''
            if i_begin == 1203:
                print(len(self.date), self.date[i_begin : i_end])
                print(len(self.sp500), self.sp500[i_begin : i_end])
                print(len(self.log_return), self.log_return[i_begin : i_end])
                print(np.std(self.log_return[i_begin : i_end], ddof = 1))
                print(factor)
                print(var)
                exit()
            '''
        self.var_10days = np.array(self.var_10days)
            
    def __comp_forward_return_10days(self):
        self.forward_return_10days = []
        for i_begin in range(1 + self.n_backtest, len(self.log_return)):
            i_end = i_begin + 10
            if i_end >= len(self.log_return):
                break

            self.forward_return_10days.append(np.log(self.sp500[i_end] / self.sp500[i_begin]))

    def read_data(self):
        with open('./data/sp500.csv', "r") as f_src:
            for i, l in enumerate(f_src):
                if i == 0:
                    continue

                l = l.strip().split(",")
                self.date.append(l[0])
                self.sp500.append(float(l[1]))
        print(f"finish reading {len(self.sp500)} data")

    def plot_histogram(self):
        figure = plt.figure(num = 1, figsize = (10, 6))
        ax = figure.add_subplot(111)
        ax.hist(self.log_return, bins = 100)
        ax.set_title("Histogram of daily log returns")
        plt.show()


if __name__ == "__main__":
    n_backtest = 21
    model = VaR_Backtest(n_backtest)
    #model.plot_histogram()
    n_breach, percent_breach, n_consecutive_breach, percent_consecutive_breach = model.comp_breach()
    #print(n_breach, percent_breach, n_consecutive_breach, percent_consecutive_breach)
    model.plot_backtest()

