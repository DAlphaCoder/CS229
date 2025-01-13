#implementation of classification algorithm
import numpy as np
import matplotlib.pyplot as plt
def main():
    # Generate random input data
    x = np.random.randn(100)  # Random array of 100 elements
    x.sort()
    print("x:", x)
    y=np.zeros(100)
    y[0:50]=1
    print("y:", y)
    logisticregression(x,y)
    newton_method(x,y)
    perceptron(x,y)

def perceptron(x,y):
    print("Perceptron")
    m=len(y)
    theta=np.array([0.0,0.0])
    alpha=1
    iterations=10000
    X=np.column_stack((np.ones(m),x))
    for _ in range(iterations):

        for i in range(m):
            h= 1 if np.dot(theta,X[i])>=0 else 0
            theta[1]=theta[1]+alpha*(y[i]-h)*x[i]
            theta[0]=theta[0]+alpha*(y[i]-h)
    print("Theta0 (Intercept):", theta[0])
    print("Theta1 (Slope):", theta[1])
    x__inpt=float(input("Enter the value of x:"))
    x__inpt=np.array([1,x__inpt])
    h=1 if np.dot(theta,x__inpt)>=0 else 0
    print("predicted value of y at x=",h)
    plot_decision_boundary(x, y, theta[0], theta[1])

def logisticregression(x,y):
    print("Logistic Regression")
    alpha=0.01
    theta0=0
    theta1=0
    m=len(x)
    iterations=1000
    h=0
    for _ in range(iterations):
        
        for i in range(m):
            h=1/(1+np.exp(-(theta0+theta1*x[i])))
            theta0=theta0-alpha*(h-y[i])
            theta1=theta1-alpha*(h-y[i])*x[i]
            
    print("Theta0 (Intercept):", theta0)
    print("Theta1 (Slope):", theta1)
    x__inpt=float(input("Enter the value of x:"))
    h=1/(1+np.exp(-(theta0+theta1*x__inpt)))
    print("predicted value of y at x=",x__inpt,"is:",0 if h<0.5 else 1)
    plot_decision_boundary(x, y, theta0, theta1)

def newton_method(x,y):
    print("Newton's Method")
    alpha=0.01
    theta=np.array([0,0])
    m=len(x)
    iterations=5
    h=0
    l=0
    X=np.array([1,x])
    gradientl=np.array([0,0])
    W=np.zeros((m,m))
    for i in m:
        gradientl+=np.dot(y[i]-h(x[i]),X)
        W[i][i]=h(x[i])*(1-h(x[i]))
    hesse=np.dot(np.dot(X,W),X.T)
    for _ in range(iterations):
        theta=theta+np.dot(np.linalg.inv(hesse),gradientl)

    print("Theta0 (Intercept):", theta[0])
    print("Theta1 (Slope):", theta[1])
    x__inpt=float(input("Enter the value of x:"))
    h=1/(1+np.exp(-(theta[0]+theta[1]*x__inpt)))
    print("predicted value of y at x=",x__inpt,"is:",0 if h<0.5 else 1)
    plot_decision_boundary(x, y, theta[0], theta[1])
def newton_method(x, y):
    print("Newton's Method")
    
    
    theta = np.array([0.0, 0.0])  
    m = len(x)  
    iterations = 5  
    
    X = np.column_stack((np.ones(m), x)) 
    
    for _ in range(iterations):
        
        h = 1 / (1 + np.exp(-np.dot(X, theta)))  
        
        gradient = np.dot(X.T, (y - h))
        
        W = np.diag(h * (1 - h))
        
        hessian = np.dot(X.T, np.dot(W, X))
        
        theta = theta + np.dot(np.linalg.inv(hessian), gradient)
    
    print("Theta0 (Intercept):", theta[0])
    print("Theta1 (Slope):", theta[1])
    
    # Predict value for a specific input
    x_input = float(input("Enter the value of x: "))
    h_input = 1 / (1 + np.exp(-(theta[0] + theta[1] * x_input)))
    print("Predicted value of y at x =", x_input, "is:", 0 if h_input < 0.5 else 1)
    
    plot_decision_boundary(x, y, theta[0], theta[1])

def plot_decision_boundary(x, y, theta0, theta1):
    plt.scatter(x, y, c=y, cmap="viridis", label="Data Points")
    decision_boundary_x = np.linspace(min(x), max(x), 100)
    decision_boundary_y = 1 / (1 + np.exp(-(theta0 + theta1 * decision_boundary_x)))
    plt.plot(decision_boundary_x, decision_boundary_y, color="red", label="Decision Boundary")
    plt.xlabel("x")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()


main()  
