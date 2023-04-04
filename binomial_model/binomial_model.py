import numpy as np
import matplotlib.pyplot as plt
from utils import *

class BinomialModel(object):
    def __init__(self, maturity, n_step, interest):
        self.reset(maturity, n_step, interest)

    def reset(self, maturity, n_step, interest):
        self.maturity = maturity
        self.n_step = n_step
        self.time_inc = self.maturity / (self.n_step - 1) # time increment (dt)
        self.interest = interest # interest rate

        self.asset_price = np.zeros((self.n_step, self.n_step))
        self.option_intrinsic_value = np.zeros((self.n_step, self.n_step))
        self.option_price = np.zeros((self.n_step, self.n_step))
        self.option_delta = np.zeros((self.n_step, self.n_step))

    def option(self, vol, spot_price, strike):
        factor_up = 1 + vol * (self.time_inc ** 0.5)
        factor_down = 1 - vol * (self.time_inc ** 0.5)
        prob_up = 0.5 + self.interest * (self.time_inc ** 0.5) / (2 * vol)
        prob_down = 1 - prob_up
        df = 1 / (1 + self.interest * self.time_inc)
        
        # binomial loop: forward
        for i_step in range(self.n_step):
            for i_node in range(i_step + 1):
                price = spot_price * (factor_down ** i_node) * (factor_up ** (i_step - i_node))
                self.asset_price[i_node, i_step] = price
                self.option_intrinsic_value[i_node, i_step] = np.maximum(price - strike, 0)

        # binomial loop: backward
        for i_step in range(self.n_step, 0, -1):
            for i_node in range(i_step):
                if i_step == self.n_step:
                    self.option_price[i_node, i_step - 1] = self.option_intrinsic_value[i_node, i_step - 1]
                    self.option_delta[i_node, i_step - 1] = 0
                else:
                    opt_prc1 = self.option_price[i_node, i_step]
                    opt_prc2 = self.option_price[i_node + 1, i_step]
                    ast_prc1 = self.asset_price[i_node, i_step]
                    ast_prc2 = self.asset_price[i_node + 1, i_step]

                    self.option_price[i_node, i_step - 1] = df * (prob_up * opt_prc1 + prob_down * opt_prc2)
                    self.option_delta[i_node, i_step - 1] = (opt_prc2 - opt_prc1) / (ast_prc2 - ast_prc1)


        self.asset_price = np.around(self.asset_price, 2)
        self.option_price = np.around(self.option_price, 2)
        self.option_delta = np.around(self.option_delta, 4)

    # Plot Price and Option Tree
    def plot_binomial_tree(self):
        plt.figure(figsize=(10, 8))
        line_sep = 0.02
        x_axis = np.array([0.02, 0.24, 0.46, 0.70, 0.90])
        y_axis = np.array([[0.58, 0.70, 0.80, 0.90, 0.90],
                              [0, 0.34, 0.60, 0.68, 0.82],
                              [0, 0,    0.24, 0.46, 0.58],
                              [0, 0,    0,    0.12, 0.36],
                              [0, 0,    0,    0,    0.12]])
        G=nx.Graph()
        for i_step in range(self.asset_price.shape[1]):
            for i_node in range(i_step + 1):
                plt.figtext(x_axis[i_step], y_axis[i_node, i_step], f'S = {self.asset_price[i_node, i_step]}')
                plt.figtext(x_axis[i_step], y_axis[i_node, i_step] - line_sep, f'V = {self.option_price[i_node, i_step]}')
                plt.figtext(x_axis[i_step], y_axis[i_node, i_step] - 2 * line_sep, f'$\Delta$ = {self.option_delta[i_node, i_step]}')

        # Plot Binomial tree
            for j in range(1, i_step + 2):
                if i_step < self.asset_price.shape[1] - 1:
                    G.add_edge((i_step, j), (i_step + 1, j))
                    G.add_edge((i_step, j), (i_step + 1, j + 1))

        posG = {}
        for node in G.nodes():
            posG[node] = (node[0], n + node[0] - 2 * node[1])
        nx.draw(G, pos = posG)

        plt.show()

if __name__ == "__main__":
    maturity = 1
    n_step = 5
    interest = 0.05
    stock_price = 100.0
    strike = 100.0

    model = BinomialModel(maturity, n_step, interest)

    vol = 0.2
    model.option(vol, stock_price, strike)
    model.plot_binomial_tree()
    exit()


    vol = np.arange(16) * 0.05 + 0.05
    option_prices = []
    for e in vol:
        model.option(e, stock_price, strike)
        option_prices.append(model.option_price[0,0])

    figure = plt.figure(num = 1, figsize = (10, 6))
    ax1 = figure.add_subplot(111)
    ax1.set_xlabel("Volatility")
    ax1.set_ylabel("Option value")
    ax1.set_xticks(vol)
    ax1.scatter(vol, option_prices)

    vol = 0.2
    n_steps = list(range(4, 51))
    option_prices = []
    for n_step in n_steps:
        model.reset(maturity, n_step, interest)
        model.option(vol, stock_price, strike)
        option_prices.append(model.option_price[0,0])

    figure2 = plt.figure(num = 2, figsize = (20, 6))
    ax2 = figure2.add_subplot(111)
    ax2.set_xlabel("Number of time steps")
    ax2.set_ylabel("Option value")
    ax2.set_xticks(list(range(4, 51, 2)))
    ax2.scatter(n_steps, option_prices)
    plt.show()

