import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

class PortfolioOptimization(object):
    def __init__(self, returns, cov, risk_free_returns):
        self.returns = returns
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)
        self.risk_free_returns = risk_free_returns
        self.n_assets = len(returns)
        self.init_weights = np.ones_like(returns) / self.n_assets

        self.bounds = [(0, 1) for i in range(self.n_assets)]

        self.min_volatility = lambda weights: np.sqrt(np.matmul(np.matmul(weights, self.cov), weights))
        self.min_variance = lambda weights: np.power(self.min_volatility(weights), 2)
        self.max_sharpe_ratio = lambda weights: -np.dot(weights, self.returns) / self.min_volatility(weights)

    def comp_efficient_frontier(self):
        all_target_returns = np.linspace(np.min(self.returns), np.max(self.returns), 200)
        all_target_vols = []

        for target_return in all_target_returns:
            constraints = ({"type" : "eq", "fun" : lambda weights: np.dot(weights, self.returns) - target_return},
                {"type" : "eq", "fun" : lambda weights: np.sum(weights) - 1})
            opt_efficient_frontier = sco.minimize(self.min_variance, self.init_weights, method = 'SLSQP', bounds = self.bounds, constraints = constraints)
            all_target_vols.append(opt_efficient_frontier["fun"])

        all_target_vols = np.array(all_target_vols)

        return all_target_returns, all_target_vols

    def comp_sharpe_portfolio_with_optimizer(self):
        constraints = ({"type" : "eq", "fun" : lambda weights: np.sum(weights) - 1})
        opt_sharpe = sco.minimize(self.max_sharpe_ratio, self.init_weights, method = 'SLSQP', constraints = constraints)
        w_tangency = opt_sharpe["x"]
        return_portfolio = np.dot(w_tangency, self.returns)
        stddev_portfolio = np.sqrt(np.matmul(np.matmul(w_tangency, self.cov), w_tangency))

        return return_portfolio, stddev_portfolio, w_tangency

    def plot_efficient_frontier(self, returns, risks, risk_free_return):
        return_tan_portfolio, stddev_tan_portfolio, _ = self.comp_sharpe_portfolio_with_formula(risk_free_return)
        #return_tan_portfolio_opt, stddev_tan_portfolio_opt, _ = self.comp_sharpe_portfolio_with_optimizer()

        figure = plt.figure(num = 1, figsize = (14, 8))
        ax = figure.add_subplot(111)
        ax.scatter(risks, returns, label = "efficient frontier")
        #ax.scatter(stddev_tan_portfolio, return_tan_portfolio, c = "red", label = "tangency portfolio, formula")
        #ax.scatter(stddev_tan_portfolio_opt, return_tan_portfolio_opt, c = "purple", label = "tangency portfolio, optimizer")
        ax.scatter(0, risk_free_return, c = "green", label = f"risk free return = {risk_free_return}")
        ax.set_xlabel("Risk (\sigma)")
        ax.set_ylabel("Return (\mu)")
        #ax.xaxis.set_ticks(np.arange(0, len(valid_date), tick_inverval))
        ax.legend()
        ax.grid(True)
        figure.savefig(f"/Users/evensong/Desktop/CQF/exams/exam1/fig/efficient_frontier_{risk_free_return}.png", format = "png")

        plt.show()

    def comp_sharpe_portfolio_with_formula(self, risk_free_return):
        w_tangency = np.matmul(self.returns - risk_free_return, self.inv_cov) / (np.dot(np.sum(self.inv_cov, 1), self.returns - risk_free_return))
        return_portfolio = np.dot(w_tangency, self.returns)
        stddev_portfolio = np.sqrt(np.matmul(np.matmul(w_tangency, self.cov), w_tangency))

        return return_portfolio, stddev_portfolio, w_tangency


if __name__ == "__main__":
    np.random.seed(1234)
    '''
    returns = np.array([0.08, 0.10, 0.10, 0.14])
    stddevs = np.eye(4) * np.array([0.12, 0.12, 0.15, 0.20])
    correl = np.array([[1, 0.2, 0.5, 0.3],
                       [0.2, 1, 0.7, 0.4],
                       [0.5, 0.7, 1, 0.9],
                       [0.3, 0.4, 0.9, 1]])
    risk_free_returns = np.array([0.05])
    plot_risk_free_returns = np.array([0.05])
    '''
    returns = np.array([0.02, 0.07, 0.15, 0.20])
    stddevs = np.eye(4) * np.array([0.05, 0.12, 0.17, 0.25])
    correl = np.array([[1, 0.3, 0.3, 0.3],
                       [0.3, 1, 0.6, 0.6],
                       [0.3, 0.6, 1, 0.6],
                       [0.3, 0.6, 0.6, 1]])
    risk_free_returns = np.array([50, 100, 150, 175]) * 1e-4
    plot_risk_free_returns = np.array([100, 175]) * 1e-4

    cov = np.matmul(np.matmul(stddevs, correl), stddevs)

    model = PortfolioOptimization(returns, cov, risk_free_returns)
    for rf in risk_free_returns:
        return_portfolio, stddev_portfolio, w_tangency = model.comp_sharpe_portfolio_with_formula(rf)
        print(f"formula: rf = {rf}, weights_portfolio = {np.round(w_tangency, 6)}, return_portfolio = {np.round(return_portfolio, 6)}, stddev_portfolio = {np.round(stddev_portfolio, 6)}")
        #return_portfolio, stddev_portfolio, w_tangency = model.comp_sharpe_portfolio_with_optimizer()
        #print(f"optimizer: rf = {rf}, weights_portfolio = {np.round(w_tangency, 6)}, return_portfolio = {np.round(return_portfolio, 6)}, stddev_portfolio = {np.round(stddev_portfolio, 6)}")

    all_target_returns, all_target_vols = model.comp_efficient_frontier()
    for rf in plot_risk_free_returns:
        model.plot_efficient_frontier(all_target_returns, all_target_vols, rf)
