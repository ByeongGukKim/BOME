import numpy as np
import sympy as sp

def comma(v, theta):
    return sp.Matrix([sp.Matrix(v), sp.Matrix(theta)])

def f(v, theta):
    return v.T*v + 2*theta.T*theta

def g(v, theta):
    return v.T*v+theta.T*theta

def grad_v_g(v, theta):
    return sp.diff(g(v,theta), v)
    
def grad_v_f(v, theta):
    return sp.diff(f(v,theta), v)

def grad_theta_f(v, theta):
    return sp.diff(f(v, theta), theta)

def grad_theta_g(v, theta):
    return sp.diff(g(v, theta), theta)

def grad_theta_theta_g(v,theta):
    return sp.diff(grad_theta_g(v, theta), theta)

def grad_theta_v_g(v,theta):
    return sp.diff(grad_theta_g(v, theta), v)

def bome(v_init, theta_init, alpha=0.01, xi=0.01, T=10, max_iter=100, eta=0.5):
    v_init = sp.MatrixSymbol('v', 10, 1)
    theta_init = sp.MatrixSymbol('theta', 9, 1)

    for k in range(max_iter):
        # Inner optimization:  approximating θ*(v) via T gradient steps
        theta_1 = theta_init - alpha * sp.diff(g(v, theta_init), theta)
        for t in range(T-1):
            theta_1 = theta_1 - alpha * sp.diff(g(v, theta_1), theta)   # Update θ using gradient of inner objective
        
        # Compute q(v, θ) and its gradient
        q = g(v, theta_init) - g(v, theta)  # q(v, θ) = g(v, θ) - g(v,θ*)
        grad_q_v = grad_v_g(v, theta_init) - grad_v_g(v, theta)
        grad_q_theta = grad_theta_g(v, theta_init) - grad_theta_g(v, theta)

        grad_q = comma(grad_q_v, grad_q_theta)
        # Compute control barrier φk
        phi_k = eta * np.linalg.norm(grad_q)**2

        # Compute the lambda_k (Lagrange multiplier)
        grad_f = comma(grad_v_f(v_init, theta_init), grad_theta_f(v_init, theta_init))
        lambda_k = max((phi_k - np.dot(grad_f, grad_q)) / (np.linalg.norm(grad_q)**2), 0)

        # Update v and θ using the BOME update rule
        v -= xi * (grad_f[0] + lambda_k * grad_q[0])
        theta -= xi * (grad_f[1] + lambda_k * grad_q[1])

        # Print the current state
        print(f"Iteration {k+1}: v = {v}, θ = {theta}, f(v, θ) = {outer_objective(v, theta)}")

    return v, theta

# Initialize variables
v_init = 0.0
theta_init = 0.0

# Run the BOME algorithm
v_opt, theta_opt = bome(v_init, theta_init)

print(f"Optimal v: {v_opt}, Optimal θ: {theta_opt}")




print(sp.Matrix(v), sp.Matrix(theta))
print(grad_theta_theta_g(v, theta))
print(grad_theta_v_g(v, theta))
print(grad_v_f(v, theta))
