# Preconditioned Gradient Descent

This program implements preconditioned gradient descent using a backtracking line search. Utilizes a preconditioner to perform a change of variables, where the gradient has smaller condition number.

The gradient descent algorithm with backtracking line search has the following exit condition:

![](https://latex.codecogs.com/gif.latex?f%28x%20&plus;%20dt%20%5CDelta%20x%29%20-%20f%28x%29%20%5Cleq%20%5Calpha%20dt%20%5Cnablaf%28x%29%5ET%20%5CDelta%20x)

Using the change of variables:

![](https://latex.codecogs.com/gif.latex?%5Cbar%7Bx%7D%20%3D%20P%5E%7B%5Cfrac%7B1%7D%7B2%7D%7Dx)

And defining the function:

![](https://latex.codecogs.com/gif.latex?%5Cbar%7Bf%7D%28x%29%3Df%28P%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7Dx%29)

One can see that the gradient search direction for minimizing our new function (f bar) is:

![](https://latex.codecogs.com/gif.latex?%5CDelta%20%5Cbar%7Bx%7D%3D-%5Cnabla%20%5Cbar%7Bf%7D%28%5Cbar%7Bx%7D%29%20%3D%20-P%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%5Cnabla%20f%28P%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%5Cbar%7Bx%7D%29%20%3D%20-P%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%5Cnabla%20f%28x%29)

This gradient search direction corresponds to the direction (in the original coordinates/function):

https://latex.codecogs.com/gif.latex?%5CDelta%20x%20%3D%20P%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%28%20-P%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%5Cnabla%20f%28x%29%29%20%3D%20-P%5E%7B-1%7D%20%5Cnabla%20f%28x%29

Giving us our update rule for preconditioned gradient descent. :)
