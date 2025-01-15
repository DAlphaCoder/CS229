#We used linear algebra to solve the linear regression problem in l1
import numpy as np
import numpy.linalg as LA
def main():
    x=np.random.randn(100,2)
    print("x:", x)
    y=100*x[:,0]+1
    print("y:", y)
    theta=LA.inv(x.T@x)@(x.T)@y
    print("theta:", theta)
main()
