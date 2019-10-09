import numpy as np
###############################################
##
# Author: Matthew Buchholz
# Date: 10/08/2019
# 
# pgd.py - script to implement preconditioned gradient descent on 
#		   a affine function composed with the softplus
##
###############################################

def pgd(x_0, f, gradient, P_inv, alpha, beta, max_iters=2e3):
	'''
			Function to implement preconditioned gradient descent with
			backtracking line search (with parameters alpha and beta)

			Args:
				x_0: numpy.array, initial condition for the descent 
				f: Python function, function to be minimized
				gradient: Python function, analytic gradient of the objective function
				P_inv: numpy.array, the (inverse of the) preconditioner
				alpha: float, fraction of the decrease in f we accept from the linear interpolation
				beta: float, backtracking parameter, the factor to reduce the step size, dt, each iteration
				max_iters: float, maximum number of iterations

			Returns:
				iters: int, number of iterations completerd
				x: numpy.array, minimizer that the algorithm finds
				np.linalg.norm(grad): float, L2 norm of the gradient upon exit condition
				f(x): float, the value of the objective function (min), upon completion
		'''
	iters = 0

	grad = gradient(x_0)
	pre  = P_inv.dot(grad)

	x = x_0
	while iters < max_iters and np.linalg.norm(grad) > 1e-7:
		iters += 1

		dt = 1
		x_t = x - dt * pre

		while f(x) - f(x_t) < alpha * dt * pre.dot(grad):
			dt *= beta
			x_t = x - dt * pre

		x = x_t
		grad = gradient(x)
		pre  = P_inv.dot(grad)

	return iters, x, np.linalg.norm(grad), f(x)

def aff_sp_grad(x):
	d1 = np.array([1, 1])
	d2 = np.array([20, 1])
	c1, c2 = 1, 2

	assert(x.shape == d1.shape)

	x1 = d1.dot(x) + c1
	x2 = d2.dot(x) + c2

	return d1*(2*logistic(x1) - 1) + d2*(2*logistic(x2) - 1)

def affine_softplus(x):
	d1 = np.array([1, 1])
	d2 = np.array([20, 1])
	c1, c2 = 1, 2

	assert(x.shape == d1.shape)

	x1 = d1.dot(x) + c1
	x2 = d2.dot(x) + c2

	return (softplus(x1) + softplus(-x1)) + (softplus(x2) + softplus(-x2))

def softplus(x):
	return np.log(1 + np.exp(x))

def logistic(x):
	return 1/(1 + np.exp(-x))

def quadratic(x):
	return x**2

def quad_grad(x):
	return 2*x

if __name__ == "__main__":
	# for quadratic minimization
	#x_0 = np.array([300])
	#P_list = [np.array([[1]]), np.array([[0.05]]), np.array([[10]])]

	# for softplus minimization
	x_0 = np.array([5, 10])

	# list of pre-conditioning matrices
	P_list = [np.identity(2), 
			  np.array([[0.05, 0],
				  		[0, 1.05]]),
			  np.array([[1.05, 0],
				  		[0, 0.05]])]

	alpha = 0.5
	beta  = 0.9

	for P_inv in P_list:
		iters, x_t, grad_norm, f_xt = pgd(x_0, affine_softplus, aff_sp_grad, P_inv, alpha, beta)
		#iters, x_t, grad_norm, f_xt = pgd(x_0, quadratic, quad_grad, P_inv, alpha, beta)
		print(iters)
		print(x_t)
		print(grad_norm)
		print(f_xt)
		print("----------------")
