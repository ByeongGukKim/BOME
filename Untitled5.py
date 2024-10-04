import numpy as np

# Outer objective function f(v, θ)
def outer_objective(v, theta):
    # 예시: f(v, θ) = (v-2)**2 + (theta-3)**2
    return (v - 2)**2 + (theta - 3)**2

# Inner objective function g(v, θ)
def inner_objective(v, theta):
    # 예시: g(v, θ) = (v + theta - 1)**2
    return (v + theta - 1)**2

# Gradient of outer objective f(v, θ)
def grad_outer(v, theta):
    df_dv = 2 * (v - 2)
    df_dtheta = 2 * (theta - 3)
    return np.array([df_dv, df_dtheta])

# Gradient of inner objective g(v, θ)
def grad_inner(v, theta):
    dg_dv = 2 * (v + theta - 1)
    dg_dtheta = 2 * (v + theta - 1)
    return np.array([dg_dv, dg_dtheta])

# Bilevel Optimization Made Easy (BOME)
def bome(v_init, theta_init, alpha=0.01, xi=0.01, T=10, max_iter=100, eta=0.5):
    v, theta = v_init, theta_init

    for k in range(max_iter):
        # Inner optimization: approximating θ*(v) via T gradient steps
        for t in range(T):
            theta -= alpha * grad_inner(v, theta)[1]  # Update θ using gradient of inner objective
        
        # Compute q(v, θ) and its gradient
        q = inner_objective(v, theta_init) - inner_objective(v, theta)  # q(v, θ) = g(v, θ) - g*(v)
        grad_q = grad_inner(v, theta)

        # Compute control barrier φk
        phi_k = eta * np.linalg.norm(grad_q)**2

        # Compute the lambda_k (Lagrange multiplier)
        grad_f = grad_outer(v, theta)
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
