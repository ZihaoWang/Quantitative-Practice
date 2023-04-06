import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

returns = np.array([0.02, 0.07, 0.15, 0.20])
stddev = np.eye(4) * np.array([0.05, 0.12, 0.17, 0.25])
correl = np.array([[1, 0.3, 0.3, 0.3],
              [0.3, 1, 0.6, 0.6],
              [0.3, 0.6, 1, 0.6],
              [0.3, 0.6, 0.6, 1]])
cov = np.matmul(np.matmul(stddev, correl), stddev)
inv_cov = np.linalg.inv(cov)
A = np.sum(inv_cov)
B = np.sum(np.matmul(returns, inv_cov))
C = np.matmul(np.matmul(returns, inv_cov), inv_cov)

def generate_data():
    weights = (np.random.random((700, 3)) - 0.5) * 200 # (-100, 100)
    weights = np.concatenate([weights, 1 - np.sum(weights, 1, keepdims = True)], 1)
    
    return weights

def plot_return_risk(returns, risks):
    figure = plt.figure(num = 1, figsize = (14, 8))
    ax = figure.add_subplot(111)
    ax.scatter(risks, returns)
    ax.set_xlabel("Risk (\sigma)")
    ax.set_ylabel("Return (\mu)")
    #ax.xaxis.set_ticks(np.arange(0, len(valid_date), tick_inverval))
    #ax.legend()
    ax.grid(True)
    figure.savefig("/Users/evensong/Desktop/CQF/exams/exam1/fig/inverse_optimization.png", format = "png")
    
    plt.show()

def inverse_optimization():
    weights = generate_data()
    global returns, cov
    returns = np.matmul(weights, returns)
    risks = np.sqrt(np.sum(np.matmul(weights, cov) * weights, 1))

    plot_return_risk(returns, risks)

def tangency_portfolio():
    global returns, cov

    risk_free_return = np.array([50, 100, 150, 175]) * 1e-4
    for i in range(len(risk_free_return)):
        rf = risk_free_return[i]
        w_tangency = np.round(np.matmul(returns - rf, inv_cov) / (B - A * rf), 6)
        stddev_pfl = np.round(np.sqrt(np.matmul(np.matmul(w_tangency, cov), w_tangency)), 6)
        print(f"rf = {rf}, w_tangency = {w_tangency}, stddev_pfl = {stddev_pfl}")

        if i == 1 or i == 3:
            weights = generate_data()
            returns = rf + np.matmul(weights, returns - rf)
            risks = np.sqrt(np.sum(np.matmul(weights, cov) * weights, 1))

            plot_return_risk(returns, risks)
            exit()




if __name__ == "__main__":
    np.random.seed(1234)
    #print(A, B, C)
    #inverse_optimization()
    tangency_portfolio()
