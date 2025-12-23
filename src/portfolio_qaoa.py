# src/portfolio_qaoa.py
"""
Portfolio optimization via QAOA using qiskit_optimization QuadraticProgram.
This solves a QUBO: maximize mu^T x - lambda x^T C x subject to cardinality constraint.
Compatible with Qiskit 1.x (uses StatevectorSampler-based QAOA).
"""

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# QAOA and optimizer imports compatible with modern qiskit-algorithms
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA

# Use StatevectorSampler for simulation
from qiskit.primitives import StatevectorSampler


def portfolio_qubo(expected_returns, cov_matrix, cardinality=3, risk_aversion=1.0, reps=1):
    n = len(expected_returns)

    # ------------------------------
    # Build the QUBO
    # ------------------------------
    qp = QuadraticProgram()

    # Binary variables x_i in {0,1}
    for i in range(n):
        qp.binary_var(name=f"x_{i}")

    # Objective: maximize mu^T x - risk_aversion * x^T C x
    # But QuadraticProgram is minimization → minimize negative
    linear = {f"x_{i}": -expected_returns[i] for i in range(n)}
    quadratic = {}

    for i in range(n):
        for j in range(n):
            quadratic[(f"x_{i}", f"x_{j}")] = risk_aversion * cov_matrix[i, j]

    qp.minimize(linear=linear, quadratic=quadratic)

    # Cardinality constraint: sum x_i == cardinality
    qp.linear_constraint(
        linear={f"x_{i}": 1 for i in range(n)},
        sense="==",
        rhs=cardinality,
        name="card"
    )

    # ------------------------------
    # QAOA Setup using StatevectorSampler
    # ------------------------------
    optimizer = COBYLA(maxiter=50)  # Reduced iterations for faster demo
    sampler = StatevectorSampler()

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps
    )

    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(qp)

    # ------------------------------
    # Extract final solution
    # ------------------------------
    x = np.array([int(result.x[i]) for i in range(n)])

    expected_ret = float(expected_returns @ x)
    variance = float(x @ cov_matrix @ x)
    score = expected_ret - risk_aversion * variance

    return {
        "selection": x,
        "expected_return": expected_ret,
        "variance": variance,
        "score": score,
        "raw_result": result
    }
