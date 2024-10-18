import numpy as np
import sympy as sp

def comma(v, theta):
    return sp.Matrix([sp.Matrix(v), sp.Matrix(theta)])

def f(v, theta):
    return v.T*v + 2*theta.T*theta

def g(v, theta):
    return v.T*v+theta.T*theta

def grad_v_g(v, theta, m):
    return sp.Matrix([sp.diff(g(v,theta)[0,0], v[i, 0]) for i in range(m)])
    
def grad_v_f(v, theta, n):
    return sp.Matrix([sp.diff(f(v,theta)[0,0], v[i, 0]) for i in range(n)])

def grad_theta_f(v, theta, n):
    return sp.Matrix([sp.diff(g(v,theta)[0,0], theta[i, 0]) for i in range(n)])

def grad_theta_g(v, theta, m):
    return sp.Matrix([sp.diff(g(v,theta)[0,0], theta[i, 0]) for i in range(m)])
    
v = sp.MatrixSymbol('v', 10, 1)
theta = sp.MatrixSymbol('theta', 9, 1)
for i in range(10):
    theta = theta - 0.01*grad_theta_g(v, theta, 9)
