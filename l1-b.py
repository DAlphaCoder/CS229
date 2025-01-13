#Using batch gradient descent to fit a line to data points

import numpy as np
import matplotlib.pyplot as plt

def main():
    x = np.random.randn(100)  
    print("x:", x)
    
    y = 100*x+1+np.random.randn(100)  # y = 2x + 1 + noise
    print("y:", y)

    theta0 = 0  # Intercept
    theta1 = 0  # Slope
    alpha = 0.001# Learning rate

    iterations = 90000
    m = len(y) 

    for _ in range(iterations):
        h = theta0 + theta1 * x
        #alpha=alpha/1.00001 , if alpha becomes small per iteration , lesser iterations are needed
        # Compute gradients manually
        grad_theta0 = h[0]-y[0]
        grad_theta1 = (h[0]-y[0]) * x[0]
        i=1
        for i in range(m):
            grad_theta0 += (h[i] - y[i])
            grad_theta1 += (h[i] - y[i]) * x[i]


        # Update parameters
        theta0 -= alpha * grad_theta0
        theta1 -= alpha * grad_theta1

    # Output final parameters
    print("Theta0 (Intercept):", theta0)
    print("Theta1 (Slope):", theta1)

    # Plot x vs y
    plt.scatter(x, y, color="blue", label="Data Points")
    
    # Plot the hypothesis line
    h_line = theta0 + theta1 * x
    plt.plot(x, h_line, color="red", label="h(x) = θ₀ + θ₁x")

    # Add labels and legend
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y with Regression Line")
    plt.legend()
    plt.grid(True)
    plt.show()

main()
