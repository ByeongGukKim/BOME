{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "77d0ad66-36cc-4626-98b6-843ca830a955",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'dg_dtheta' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[7], line 58\u001b[0m\n\u001b[0;32m     55\u001b[0m theta_init \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0.0\u001b[39m\n\u001b[0;32m     57\u001b[0m \u001b[38;5;66;03m# Run the BOME algorithm\u001b[39;00m\n\u001b[1;32m---> 58\u001b[0m v_opt, theta_opt \u001b[38;5;241m=\u001b[39m bome(v_init, theta_init)\n\u001b[0;32m     60\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mOptimal v: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mv_opt\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m, Optimal θ: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mtheta_opt\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m)\n",
      "Cell \u001b[1;32mIn[7], line 31\u001b[0m, in \u001b[0;36mbome\u001b[1;34m(v_init, theta_init, alpha, xi, T, max_iter, eta)\u001b[0m\n\u001b[0;32m     28\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m k \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mrange\u001b[39m(max_iter):\n\u001b[0;32m     29\u001b[0m     \u001b[38;5;66;03m# Inner optimization:  approximating θ*(v) via T gradient steps\u001b[39;00m\n\u001b[0;32m     30\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m t \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mrange\u001b[39m(T):\n\u001b[1;32m---> 31\u001b[0m         theta \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m=\u001b[39m alpha \u001b[38;5;241m*\u001b[39m dg_dtheta[\u001b[38;5;241m1\u001b[39m]  \u001b[38;5;66;03m# Update θ using gradient of inner objective\u001b[39;00m\n\u001b[0;32m     33\u001b[0m     \u001b[38;5;66;03m# Compute q(v, θ) and its gradient\u001b[39;00m\n\u001b[0;32m     34\u001b[0m     q \u001b[38;5;241m=\u001b[39m inner_objective(v, theta_init) \u001b[38;5;241m-\u001b[39m inner_objective(v, theta)  \u001b[38;5;66;03m# q(v, θ) = g(v, θ) - g(v,θ*)\u001b[39;00m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'dg_dtheta' is not defined"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import sympy as sp\n",
    "v = sp.MatrixSymbol('v', m, 1)\n",
    "theta = sp.MatrixSymbol('theta', n, 1)\n",
    "\n",
    "# Outer objective function f(v, θ)\n",
    "def outer_objective(v, theta):\n",
    "    return sp.Trace(v.T*v + 2*theta.T*theta)\n",
    "\n",
    "# Inner objective function g(v, θ)\n",
    "def inner_objective(v, theta):\n",
    "    return sp.Trace(v.T*v*theta.T*theta)\n",
    "\n",
    "# Gradient of outer objective f(v, θ)\n",
    "def grad_outer(v, theta):\n",
    "    df_dv = sp.diff(outer_objective(v, theta), v)\n",
    "    df_dtheta = sp.diff(outer_objective(v, theta), theta)\n",
    "    return np.array([df_dv, df_dtheta])\n",
    "\n",
    "# Gradient of inner objective g(v, θ)\n",
    "def grad_inner(v, theta):\n",
    "    dg_dv = sp.diff(inner_objective(v, theta), v)\n",
    "    dg_dtheta = sp.diff(inner_objective(v, theta), theta)\n",
    "    return np.array([dg_dv, dg_dtheta])\n",
    "\n",
    "# Bilevel Optimization Made Easy (BOME)\n",
    "def bome(v_init, theta_init, alpha=0.01, xi=0.01, T=10, max_iter=100, eta=0.5):\n",
    "    v, theta = v_init, theta_init\n",
    "\n",
    "    for k in range(max_iter):\n",
    "        # Inner optimization:  approximating θ*(v) via T gradient steps\n",
    "        for t in range(T):\n",
    "            theta -= alpha * dg_dtheta  # Update θ using gradient of inner objective\n",
    "        \n",
    "        # Compute q(v, θ) and its gradient\n",
    "        q = inner_objective(v, theta_init) - inner_objective(v, theta)  # q(v, θ) = g(v, θ) - g(v,θ*)\n",
    "        grad_q = grad_inner(v, theta_init) - grad_inner(v, theta)\n",
    "\n",
    "        # Compute control barrier φk\n",
    "        phi_k = eta * np.linalg.norm(grad_q)**2\n",
    "\n",
    "        # Compute the lambda_k (Lagrange multiplier)\n",
    "        grad_f = grad_outer(v, theta)\n",
    "        lambda_k = max((phi_k - np.dot(grad_f, grad_q)) / (np.linalg.norm(grad_q)**2), 0)\n",
    "\n",
    "        # Update v and θ using the BOME update rule\n",
    "        v -= xi * (grad_f[0] + lambda_k * grad_q[0])\n",
    "        theta -= xi * (grad_f[1] + lambda_k * grad_q[1])\n",
    "\n",
    "        # Print the current state\n",
    "        print(f\"Iteration {k+1}: v = {v}, θ = {theta}, f(v, θ) = {outer_objective(v, theta)}\")\n",
    "\n",
    "    return v, theta\n",
    "\n",
    "# Initialize variables\n",
    "v_init = 0.0\n",
    "theta_init = 0.0\n",
    "\n",
    "# Run the BOME algorithm\n",
    "v_opt, theta_opt = bome(v_init, theta_init)\n",
    "\n",
    "print(f\"Optimal v: {v_opt}, Optimal θ: {theta_opt}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d18c8d3d-a0d3-47f8-82c2-6af6954d90fe",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
