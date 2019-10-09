# Preconditioned Gradient Descent

This program implements preconditioned gradient descent using a backtracking line search. Utilizes a preconditioner to perform a change of variables, where the gradient has smaller condition number.

The gradient descent algorithm with backtracking line search has the following exit condition:

![](https://latex.codecogs.com/gif.latex?f%28x%20&plus;%20dt%20%5CDelta%20x%29%20-%20f%28x%29%20%5Cleq%20%5Calpha%20dt%20%5Cnablaf%28x%29%5ET%20%5CDelta%20x)
