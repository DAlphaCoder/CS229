#locally weighted regression
import numpy as np
import matplotlib.pyplot as plt
def locallyWeightedRegression():
    
    x=np.random.rand(100)
    y=15*x**5+13*x**3+2*x+np.random.randn(100)
    print("x:", x)
    print("y:", y)
    inpx=np.random.rand()
    print("inpx:",inpx)
    tau=0.5
    weight=np.exp(-((x-inpx)**2)/(2*tau**2))
    theta0 = 0  # Intercept
    theta1 = 0  # Slope
    alpha = 0.01# Learning rate

    iterations = 900
    m = len(y)  

    for _ in range(iterations):
        # Compute hypothesis
        h = theta0 + theta1 * x
        #alpha=alpha/1.00001 for more accuracy
        # Compute gradients manually
        grad_theta0 = h[0]-y[0]
        grad_theta1 = (h[0]-y[0]) * x[0]
        i=1
        for i in range(m):
            grad_theta0 += (h[i] - y[i])*weight[i]
            grad_theta1 += (h[i] - y[i]) * x[i]*weight[i]


        # Update parameters
        theta0 -= alpha * grad_theta0
        theta1 -= alpha * grad_theta1

    # Output final parameters
    print("Theta0 (Intercept):", theta0)
    print("Theta1 (Slope):", theta1)
    print("predicted value of y at x=",inpx,"is:",theta0+theta1*inpx)
    iterations = 900
    m = len(y)  # Number of data points

    for _ in range(iterations):
        # Compute hypothesis
        h = theta0 + theta1 * x
        #alpha=alpha/1.00001
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
    print("predicted value of y at x=",inpx,"is:",theta0+theta1*inpx)
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

def main():
    locallyWeightedRegression()
main()
   
