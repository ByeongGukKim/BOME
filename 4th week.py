import numpy as np
import sympy as sp

def comma(v, theta):
    return sp.Matrix([sp.Matrix(v), sp.Matrix(theta)])

def f(v_1, theta_1):
    return v_1.T*v_1 + 2*theta_1.T*theta_1

def g(v_1, theta_1):
    return v_1.T*v_1+theta_1.T*theta_1

def grad_v_g(v, v_1, theta_1, m):
    return sp.Matrix([sp.diff(g(v_1,theta_1)[0,0], v[i, 0]) for i in range(m)])
    
def grad_v_f(v, v_1, theta_1, n):
    return sp.Matrix([sp.diff(f(v_1,theta_1)[0,0], v[i, 0]) for i in range(n)])

def grad_theta_f(v, theta, theta_1, n):
    return sp.Matrix([sp.diff(f(v_1,theta_1)[0,0], theta[i, 0]) for i in range(n)])

def grad_theta_g(v, theta, theta_1, m):
    return sp.Matrix([sp.diff(g(v_1,theta_1)[0,0], theta[i, 0]) for i in range(m)])

def bome(v_init, theta_init, alpha=0.01, xi=0.01, T=10, max_iter=100, eta=0.5, m=10, n=9):
    theta = sp.MatrixSymbol('theta', m, 1)
    v = sp.MatrixSymbol('v', n, 1)
    v_init = sp.MatrixSymbol('v', m, 1)
    theta_init = sp.MatrixSymbol('theta', n, 1)
    for k in range(max_iter):
        theta_T = theta_
        for t in range(T-1):
            theta_T = sp.Matrix(theta_T) - alpha * grad_theta_g(v, theta, theta_T, n)   # Update θ using gradient of inner objective
        
        # Compute q(v, θ) and its gradient
        q = g(v_init, theta_init) - g(v_init, theta_T)  # q(v, θ) = g(v, θ) - g(v,θ*)
        grad_q_v = grad_v_g(v, v_init, theta_init, m) - grad_v_g(v, v_init, theta_T, m)
        grad_q_theta = grad_theta_g(theta, v_init, theta_init, 9) - grad_theta_g(theta, v_init, theta_T, n)
        grad_q = comma(grad_q_v, grad_q_theta)
        # Compute control barrier φk
        phi_k = eta * grad_q.norm()**2

        # Compute the lambda_k (Lagrange multiplier)
        grad_f = comma(grad_v_f(v, v_init, theta_init, m), grad_theta_f(theta, v_init, theta_init, n))
        lambda_k = max((phi_k - grad_f.dot(grad_q) / grad_q.norm()**2), 0)

        # Update v and θ using the BOME update rule
        matrix_v_theta = comma(v_init, theta_init)
        matrix_v_theta -= 0.01*(grad_f + lambda_k * grad_q)
        v_init = matrix_v_theta[:m, :]  # 상단 m개의 행
        theta_init = matrix_v_theta[m:, :]  # 하단 n개의 행
    return matrix_v_theta
