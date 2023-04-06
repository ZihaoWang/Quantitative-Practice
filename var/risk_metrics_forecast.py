import numpy as np
import matplotlib.pyplot as plt
import scipy
from adjustText import adjust_text


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
        self.var_hist = None
        self.stddev_estimate_ahead = None
        self.all_breaches = None
        self.coeff = 0.72

        self.read_data()
        self.sp500 = np.array(self.sp500)
        self.log_return = np.array([np.log(self.sp500[i]) - np.log(self.sp500[i - 1]) for i in range(1, len(self.sp500))])
        self.log_return = np.concatenate((np.array([np.nan]), self.log_return)) # align with self.date and self.sp500
        self.squared_return = np.power(self.log_return, 2)
        self.var_estimation = np.zeros_like(self.squared_return)
        self.var_estimation[0] = np.nan
        self.var_estimation[1] = np.mean(self.squared_return[1 : 251])
        for i in range(2, len(self.var_estimation)):
            self.var_estimation[i] = self.coeff * self.var_estimation[i - 1] + (1 - self.coeff) * self.squared_return[i - 1] # EWMA estimation

        self.mu = np.mean(self.log_return[1:])
        self.sigma = np.std(self.log_return[1:])
        self.sigma_10day = self.sigma * np.sqrt(10)
        self.sigma_year = self.sigma * np.sqrt(252)
        print(f"mu = {self.mu}, 1 day sigma = {self.sigma}, 10 days sigma = {self.sigma_10day}, 1 year sigma = {self.sigma_year}")

    def comp_breach(self):
        self.__comp_var_hist()
        self.__comp_vol_forcast()
        
        assert len(self.var_hist) == len(self.stddev_estimate_ahead)

        self.all_breaches = (self.stddev_estimate_ahead > self.var_hist).astype(int)
        n_breach = np.sum(self.all_breaches)
        percent_breach = n_breach / self.all_breaches.shape[0]

        n_consecutive_breach = 0
        for i in range(0, self.all_breaches.shape[0] - 1):
            if self.all_breaches[i] == 1 and self.all_breaches[i + 1] == 1:
                n_consecutive_breach += 1
            percent_consecutive_breach = n_consecutive_breach / (self.all_breaches.shape[0] - 1)

        return n_breach, percent_breach, n_consecutive_breach, percent_consecutive_breach

    def plot_backtest(self):
        valid_date = self.date
        print(len(valid_date))
        n_tick_date = 8
        tick_inverval = len(valid_date) // n_tick_date
        tick_dates = [valid_date[i * tick_inverval] for i in range(n_tick_date)]

        figure = plt.figure(num = 1, figsize = (14, 8))
        ax = figure.add_subplot(111)
        ax.scatter(valid_date, self.var_hist)
        colors = ["orange" if e == 0 else "red" for e in self.all_breaches]
        ax.scatter(valid_date, self.stddev_estimate_ahead, c = colors)
        ax.plot(valid_date, self.var_hist, label = "10 days HistSim VaR")
        ax.plot(valid_date, self.stddev_estimate_ahead, label = "RM 10 days Vol Forecast")
        
        '''
        annotation_texts = []
        for i, is_breach in enumerate(self.all_breaches):
            if is_breach == 1:
                annotation_texts.append(plt.text(valid_date[i], self.stddev_estimate_ahead[i], valid_date[i], {"c" : "red"}, ha = "left", va = "bottom"))
        adjust_text(annotation_texts, force_text=(5, 20), force_explode = (7, 40), min_arrow_len = 10, arrowprops=dict(arrowstyle='->', color='red'))
        '''
        ax.set_title("RiskMetrics Forecast v.s. Hist Sim VaR")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (%)")
        ax.xaxis.set_ticks(np.arange(0, len(valid_date), tick_inverval))
        ax.legend()
        ax.grid(True)
        figure.savefig("/Users/evensong/Desktop/CQF/exams/exam1/fig/RiskMetric_forcast.png", format = "png")
        
        plt.show()

    def __comp_var_hist(self, confidence = 0.99):
        self.var_hist = [np.nan] * self.n_backtest
        perc = (1 - confidence) * 100

        for i_begin in range(1, len(self.var_estimation)):
            i_end = i_begin + self.n_backtest
            if i_end > len(self.log_return):
                break

            var = -np.percentile(self.log_return[i_begin : i_end], perc) * np.sqrt(10) # because EWMA prediction is positive, we use negative VaR here for comparison
            self.var_hist.append(var)
        self.var_hist = np.array(self.var_hist)
            
    def __comp_vol_forcast(self):
        self.stddev_estimate_ahead = np.sqrt(self.var_estimation * 10)

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
    print(n_breach, percent_breach, n_consecutive_breach, percent_consecutive_breach)
    model.plot_backtest()

