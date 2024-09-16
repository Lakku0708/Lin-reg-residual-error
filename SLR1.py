import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

# Number of data points
N = len(x)


# 1. Compute Regression Coefficients using Analytic Formulation (Normal Equation)
def analytic_solution(x, y):
    # Mean of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Compute coefficients
    numerator = N * np.sum(x * y) - np.sum(x) * np.sum(y)
    denominator = N * np.sum(x ** 2) - (np.sum(x)) ** 2
    beta_1 = numerator / denominator
    beta_0 = np.mean(y) - beta_1 * np.mean(x)

    return beta_0, beta_1


# 2. Compute SSE and R^2 value
def compute_sse_r2(beta_0, beta_1, x, y):
    y_pred = beta_0 + beta_1 * x
    sse = np.sum((y - y_pred) ** 2)
    total_variance = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (sse / total_variance)
    return sse, r2


# 3. Gradient Descent (Full-Batch)
def gradient_descent(x, y, alpha=0.01, iterations=1000):
    beta_0, beta_1 = 0.0, 0.0
    N = len(x)

    for _ in range(iterations):
        y_pred = beta_0 + beta_1 * x
        error = y_pred - y
        beta_0 -= alpha * (1 / N) * np.sum(error)
        beta_1 -= alpha * (1 / N) * np.sum(error * x)

    return beta_0, beta_1


# 4. Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(x, y, alpha=0.01, iterations=100):
    beta_0, beta_1 = 0.0, 0.0
    N = len(x)

    for _ in range(iterations):
        for i in range(N):
            y_pred = beta_0 + beta_1 * x[i]
            error = y_pred - y[i]
            beta_0 -= alpha * error
            beta_1 -= alpha * error * x[i]

    return beta_0, beta_1
# Analytic solution
beta_0_analytic, beta_1_analytic = analytic_solution(x, y)
sse_analytic, r2_analytic = compute_sse_r2(beta_0_analytic, beta_1_analytic, x, y)
# Gradient Descent solution (Full-batch)
beta_0_gd, beta_1_gd = gradient_descent(x, y)
sse_gd, r2_gd = compute_sse_r2(beta_0_gd, beta_1_gd, x, y)

# Stochastic Gradient Descent solution (SGD)
beta_0_sgd, beta_1_sgd = stochastic_gradient_descent(x, y)
sse_sgd, r2_sgd = compute_sse_r2(beta_0_sgd, beta_1_sgd, x, y)

# Results
print("Analytic Solution: beta_0 =", beta_0_analytic, ", beta_1 =", beta_1_analytic)
print("SSE (Analytic):", sse_analytic, ", R^2 (Analytic):", r2_analytic)

print("\nGradient Descent Solution: beta_0 =", beta_0_gd, ", beta_1 =", beta_1_gd)
print("SSE (GD):", sse_gd, ", R^2 (GD):", r2_gd)

print("\nStochastic Gradient Descent Solution: beta_0 =", beta_0_sgd, ", beta_1 =", beta_1_sgd)
print("SSE (SGD):", sse_sgd, ", R^2 (SGD):", r2_sgd)

# Plotting the results
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, beta_0_analytic + beta_1_analytic * x, color='red', label='Analytic Solution')
plt.plot(x, beta_0_gd + beta_1_gd * x, color='green', label='Gradient Descent')
plt.plot(x, beta_0_sgd + beta_1_sgd * x, color='orange', label='Stochastic GD')
plt.legend()
plt.show()
