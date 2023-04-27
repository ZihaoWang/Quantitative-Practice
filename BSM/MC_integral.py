import numpy as np
import matplotlib.pyplot as plt

class FiniteDifference1D(object):
    def __init__(self, x_range, y_range):
        self.x_begin, self.x_end = x_range[0], x_range[1]# + 1e-6
        self.y_begin, self.y_end = y_range[0], y_range[1]# + 1e-6
        self.P = 3
        self.Q = 2
        self.f = lambda x: 4 * x * x

    def solve_exact(self, n_step):
        x_mesh = np.linspace(self.x_begin, self.x_end, n_step + 1)
        a = (np.exp(3) * 17 - np.exp(1) * 12) / (np.exp(1) - 1)
        b = (np.exp(3) * 12 - np.exp(4) * 17) / (np.exp(1) - 1)

        y_mesh = np.zeros(n_step + 1)
        y_mesh[0], y_mesh[-1] = self.y_begin, self.y_end
        for i in range(1, x_mesh.shape[0] - 1):
            x = x_mesh[i]
            y_mesh[i] = a * np.exp(-x) + b * np.exp(-2 * x) + 2 * x * x - 6 * x - 7

        return x_mesh, y_mesh
            
    
    def solve_fd(self, n_step):
        x_mesh = np.linspace(self.x_begin, self.x_end, n_step + 1)
        dx = (self.x_end - self.x_begin) / n_step

        r = 1 - 0.5 * dx * self.P
        s = -2 + dx * dx * self.Q
        t = 1 + 0.5 * dx * self.P

        A = np.zeros((n_step + 1, n_step + 1))
        b = x_mesh
        b = self.f(b) * dx * dx
        b[0] = self.y_begin
        b[-1] = self.y_end

        A[0, 0] = A[-1, -1] = 1
        for i in range(1, A.shape[0] - 1):
            A[i, i - 1] = r
            A[i, i] = s
            A[i, i + 1] = t
        
        y_mesh = np.linalg.solve(A, b)

        return x_mesh, y_mesh



if __name__ == "__main__":
    np.set_printoptions(precision = 3, suppress = True)
    x_range = (1, 2)
    y_range = (1, 6)

    fd = FiniteDifference1D(x_range, y_range)
    x1, y1 = fd.solve_exact(100)
    figure = plt.figure(num = 1, figsize = (14, 8))
    ax = figure.add_subplot(111)    
    ax.plot(x1, y1, label = "Exact Solution", color = "orange")
    n_steps = [10, 50, 100]
    linestyles = [":", "--", "-"]
    colors = ["green", "blue", "cyan"]
    for i, n_step in enumerate(n_steps):
        x2, y2 = fd.solve_fd(n_step)

        if i == 0:
            print(f"x = {x2}")
            print(f"With FD, y = {y2}")
        ax.plot(x2, y2, label = f"FD method, step = {n_step}", linestyle = linestyles[i], color = colors[i])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fname = "fd"
    figure.savefig(f"/Users/evensong/Desktop/CQF/exams/exam2/fig/{fname}.png", format = "png")

    

