#most basic linear regression model using stochastic gradient descent
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Generate random input data
    x = np.random.randn(10)  # Random array of 10 elements
    print("x:", x)
    
    # Generate target values as random integers between 0 and 10
   #y = np.random.randint(0, 10, 10)
    y = 2*x+1+np.random.randn(10)  # y = 2x + 1 + noise 
    print("y:", y)

    # Initialize parameters
    theta0 = 0  # Intercept
    theta1 = 0  # Slope
    alpha = 0.01  # Learning rate

    # Perform stochastic gradient descent
    iterations = 1000
    m = len(y)  # Number of data points

    for _ in range(iterations):
        for i in range(m):  # Update for each data point
            # Compute hypothesis for a single data point
            h = theta0 + theta1 * x[i]

            # Compute gradients for the current data point
            grad_theta0 = h - y[i]
            grad_theta1 = (h - y[i]) * x[i]

            # Update parameters
            theta0 -= alpha * grad_theta0
            theta1 -= alpha * grad_theta1

    # Output final parameters
    print("Theta0 (Intercept):", theta0)
    print("Theta1 (Slope):", theta1)

    #Plot x vs y
    plt.scatter(x, y, color="blue", label="Data Points")
    
    #Plot the hypothesis line
    h_line = theta0 + theta1 * x
    plt.plot(x, h_line, color="red", label="h(x) = θ₀ + θ₁x")

    #Add labels and legend
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x vs y with Regression Line (SGD)")
    plt.legend()
    plt.grid(True)
    plt.show()

main()
