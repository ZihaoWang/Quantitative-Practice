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

if __name__ == "__main__":
    np.random.seed(1234)
    inverse_optimization()
