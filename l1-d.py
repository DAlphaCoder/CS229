#USing multiple features for linear regression using Stochastic Gradient Descent
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Generate random input data with 3 features (10 samples, 3 features)
    x = np.random.randn(10, 3)  # Random array of shape (10, 3)
    print("x:\n", x)
    
    # Generate target values as random integers between 0 and 10
    y = np.random.randint(0, 10, 10)
    print("y:", y)

    # Initialize parameters
    theta0 = 0  # Intercept
    theta = np.zeros(3)  # Coefficients for x1, x2, x3
    alpha = 0.01  # Learning rate

    # Perform stochastic gradient descent
    iterations = 1000
    m = len(y)  # Number of data points

    for _ in range(iterations):
        for i in range(m):  # Update for each data point
            # Compute hypothesis for a single data point
            h = theta0 + np.dot(theta, x[i])  # h = θ0 + θ1*x1 + θ2*x2 + θ3*x3

            # Compute gradients for the current data point
            grad_theta0 = h - y[i]  # Gradient for θ0
            grad_theta = (h - y[i]) * x[i]  # Gradient for θ1, θ2, θ3

            # Update parameters
            theta0 -= alpha * grad_theta0
            theta -= alpha * grad_theta

    # Output final parameters
    print("Theta0 (Intercept):", theta0)
    print("Theta (Slopes):", theta)

    # Plot x1 vs y and show h(x)
    plt.scatter(x[:, 0], y, color="blue", label="Data Points (x1 vs y)")
    
    # Compute predictions for plotting
    h_line = theta0 + np.dot(x, theta)
    plt.plot(x[:, 0], h_line, color="red", label="h(x) = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₃")

    # Add labels and legend
    plt.xlabel("x1 (Feature 1)")
    plt.ylabel("y")
    plt.title("x1 vs y with Regression Line (SGD)")
    plt.legend()
    plt.grid(True)
    plt.show()

main()
