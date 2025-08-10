"""
Advanced Control Methods Implementation and Examples

This file implements the core concepts from 03_advanced_control.md:

1. Finite-Horizon MDPs: Dynamic programming for time-limited problems
2. Linear Quadratic Regulation (LQR): Optimal control for linear systems
3. Linearization: Approximating nonlinear dynamics with linear models
4. Differential Dynamic Programming (DDP): Iterative trajectory optimization
5. Linear Quadratic Gaussian (LQG): Optimal control under uncertainty
6. Kalman Filtering: State estimation for partially observable systems

Key Concepts Demonstrated:
- Finite-horizon Bellman equation: V_t(s) = max_a [R_t(s,a) + Σ P(s'|s,a) V_{t+1}(s')]
- LQR optimal control: u_t = -L_t x_t
- Linearization: F(x,u) ≈ A(x-x₀) + B(u-u₀) + c
- DDP: Iterative linearization and LQR solution
- LQG: Separation principle and Kalman filtering

"""

import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Dict, Optional
from dataclasses import dataclass

# ----------------------
# Finite-Horizon MDPs
# ----------------------

def finite_horizon_value_iteration(states: List[int], actions: List[int], 
                                 P: Callable, R: Callable, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value iteration for finite-horizon MDPs.
    
    This implements the finite-horizon Bellman equation:
    V_t(s) = max_a [R_t(s,a) + Σ P(s'|s,a) V_{t+1}(s')]
    
    The algorithm works backwards from the terminal time T:
    1. Set V_T(s) = max_a R_T(s,a) for all states
    2. For t = T-1, T-2, ..., 0:
       V_t(s) = max_a [R_t(s,a) + Σ P(s'|s,a) V_{t+1}(s')]
       π_t(s) = argmax_a [R_t(s,a) + Σ P(s'|s,a) V_{t+1}(s')]
    
    Args:
        states: List of state indices
        actions: List of action indices
        P: Function P(t, s, a, s_next) -> probability
        R: Function R(t, s, a) -> reward
        T: Time horizon (int)
        
    Returns:
        V: Value function array of shape (T+1, n_states)
        pi: Policy array of shape (T, n_states)
    """
    nS = len(states)
    nA = len(actions)
    
    # Initialize value function and policy arrays
    V = np.zeros((T+1, nS))
    pi = np.zeros((T, nS), dtype=int)
    
    # Terminal value function: V_T(s) = max_a R_T(s,a)
    for s in states:
        V[T, s] = max(R(T, s, a) for a in actions)
    
    # Backward induction: solve for V_t and π_t
    for t in reversed(range(T)):
        for s in states:
            # Compute Q-values for all actions
            Q_values = []
            for a in actions:
                # Expected value: R(t,s,a) + Σ P(s'|s,a) V_{t+1}(s')
                expected_value = 0.0
                for s_next in states:
                    expected_value += P(t, s, a, s_next) * V[t+1, s_next]
                Q_values.append(R(t, s, a) + expected_value)
            
            # Take maximum over actions
            V[t, s] = max(Q_values)
            pi[t, s] = np.argmax(Q_values)
    
    return V, pi

def finite_horizon_policy_evaluation(states: List[int], actions: List[int],
                                   P: Callable, R: Callable, pi: np.ndarray) -> np.ndarray:
    """
    Policy evaluation for finite-horizon MDPs.
    
    This solves the finite-horizon Bellman equation for a given policy:
    V_t^π(s) = R_t(s,π_t(s)) + Σ P(s'|s,π_t(s)) V_{t+1}^π(s')
    
    Args:
        states: List of state indices
        actions: List of action indices
        P: Transition probability function
        R: Reward function
        pi: Policy array of shape (T, n_states)
        
    Returns:
        V: Value function array of shape (T+1, n_states)
    """
    nS = len(states)
    T = pi.shape[0]
    
    V = np.zeros((T+1, nS))
    
    # Terminal value (can be set to zero or specified)
    V[T, :] = 0.0
    
    # Backward induction
    for t in reversed(range(T)):
        for s in states:
            a = pi[t, s]
            expected_value = 0.0
            for s_next in states:
                expected_value += P(t, s, a, s_next) * V[t+1, s_next]
            V[t, s] = R(t, s, a) + expected_value
    
    return V

# ----------------------
# Linear Quadratic Regulation (LQR)
# ----------------------

def discrete_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, 
                T: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Finite-horizon discrete-time LQR via dynamic programming.
    
    This solves the optimal control problem:
    min_u Σ_{t=0}^{T-1} [x_t^T Q x_t + u_t^T R u_t] + x_T^T Q_T x_T
    subject to: x_{t+1} = A x_t + B u_t
    
    The optimal control law is: u_t = -L_t x_t
    where L_t is the feedback gain matrix.
    
    Args:
        A: State transition matrix (n x n)
        B: Control input matrix (n x m)
        Q: State cost matrix (n x n)
        R: Control cost matrix (m x m)
        T: Time horizon
        
    Returns:
        L: List of feedback gains L_t for t = 0, 1, ..., T-1
        P: List of value function matrices P_t for t = 0, 1, ..., T
    """
    n = A.shape[0]  # State dimension
    m = B.shape[1]  # Control dimension
    
    # Initialize arrays
    P = [None] * (T+1)
    L = [None] * T
    
    # Terminal cost matrix
    P[T] = Q.copy()
    
    # Backward recursion
    for t in reversed(range(T)):
        # Compute optimal feedback gain
        # L_t = (B^T P_{t+1} B + R)^(-1) B^T P_{t+1} A
        BtPB = B.T @ P[t+1] @ B
        BtPA = B.T @ P[t+1] @ A
        
        L[t] = np.linalg.solve(BtPB + R, BtPA).T
        
        # Update value function matrix
        # P_t = Q + A^T P_{t+1} (A - B L_t)
        P[t] = Q + A.T @ P[t+1] @ (A - B @ L[t])
    
    return L, P

def continuous_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infinite-horizon continuous-time LQR.
    
    This solves the algebraic Riccati equation:
    A^T P + P A - P B R^(-1) B^T P + Q = 0
    
    The optimal control law is: u = -L x
    where L = R^(-1) B^T P
    
    Args:
        A: System matrix (n x n)
        B: Input matrix (n x m)
        Q: State cost matrix (n x n)
        R: Input cost matrix (m x m)
        
    Returns:
        L: Optimal feedback gain matrix
        P: Solution of algebraic Riccati equation
    """
    # Solve continuous-time algebraic Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    
    # Compute optimal feedback gain
    L = np.linalg.solve(R, B.T @ P)
    
    return L, P

def simulate_lqr_system(A: np.ndarray, B: np.ndarray, L: np.ndarray, 
                       x0: np.ndarray, T: int, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a system under LQR control.
    
    Args:
        A: System matrix
        B: Input matrix
        L: Feedback gain matrix
        x0: Initial state
        T: Simulation time
        dt: Time step
        
    Returns:
        t: Time array
        x: State trajectory
    """
    n_steps = int(T / dt)
    n_states = A.shape[0]
    
    t = np.linspace(0, T, n_steps)
    x = np.zeros((n_steps, n_states))
    x[0] = x0
    
    # Forward Euler integration
    for i in range(n_steps - 1):
        u = -L @ x[i]  # Optimal control law
        dx = A @ x[i] + B @ u  # System dynamics
        x[i+1] = x[i] + dt * dx
    
    return t, x

# ----------------------
# Linearization of Nonlinear Dynamics
# ----------------------

def linearize_dynamics(F: Callable, s0: np.ndarray, a0: np.ndarray, 
                      eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearize a nonlinear function F(s, a) around (s0, a0).
    
    This computes the first-order Taylor approximation:
    F(s, a) ≈ A(s - s0) + B(a - a0) + c
    
    where:
    - A = ∂F/∂s evaluated at (s0, a0)
    - B = ∂F/∂a evaluated at (s0, a0)
    - c = F(s0, a0)
    
    Args:
        F: Function F(s, a) -> s_next
        s0: Nominal state
        a0: Nominal action
        eps: Finite difference step size
        
    Returns:
        A: State Jacobian matrix
        B: Action Jacobian matrix
        c: Constant term
    """
    s0 = np.atleast_1d(s0)
    a0 = np.atleast_1d(a0)
    n = s0.size  # State dimension
    m = a0.size  # Action dimension
    
    # Evaluate function at nominal point
    f0 = F(s0, a0)
    
    # Initialize Jacobian matrices
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    
    # Compute state Jacobian A = ∂F/∂s
    for i in range(n):
        ds = np.zeros(n)
        ds[i] = eps
        A[:, i] = (F(s0 + ds, a0) - f0) / eps
    
    # Compute action Jacobian B = ∂F/∂a
    for j in range(m):
        da = np.zeros(m)
        da[j] = eps
        B[:, j] = (F(s0, a0 + da) - f0) / eps
    
    # Constant term
    c = f0 - A @ s0 - B @ a0
    
    return A, B, c

def linearize_cost_function(cost_fn: Callable, s0: np.ndarray, a0: np.ndarray,
                          eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Linearize a cost function around (s0, a0).
    
    This computes the quadratic approximation:
    C(s, a) ≈ (s-s0)^T Q (s-s0) + (a-a0)^T R (a-a0) + (s-s0)^T S (a-a0) + c
    
    Args:
        cost_fn: Function C(s, a) -> cost
        s0: Nominal state
        a0: Nominal action
        eps: Finite difference step size
        
    Returns:
        Q: State cost matrix
        R: Action cost matrix
        S: Cross term matrix
        c: Constant term
    """
    s0 = np.atleast_1d(s0)
    a0 = np.atleast_1d(a0)
    n = s0.size
    m = a0.size
    
    # Evaluate cost at nominal point
    c0 = cost_fn(s0, a0)
    
    # Initialize matrices
    Q = np.zeros((n, n))
    R = np.zeros((m, m))
    S = np.zeros((n, m))
    
    # Compute second derivatives using finite differences
    for i in range(n):
        for j in range(n):
            ds1 = np.zeros(n)
            ds2 = np.zeros(n)
            ds1[i] = eps
            ds2[j] = eps
            
            Q[i, j] = (cost_fn(s0 + ds1 + ds2, a0) - cost_fn(s0 + ds1, a0) - 
                      cost_fn(s0 + ds2, a0) + c0) / (eps * eps)
    
    for i in range(m):
        for j in range(m):
            da1 = np.zeros(m)
            da2 = np.zeros(m)
            da1[i] = eps
            da2[j] = eps
            
            R[i, j] = (cost_fn(s0, a0 + da1 + da2) - cost_fn(s0, a0 + da1) - 
                      cost_fn(s0, a0 + da2) + c0) / (eps * eps)
    
    for i in range(n):
        for j in range(m):
            ds = np.zeros(n)
            da = np.zeros(m)
            ds[i] = eps
            da[j] = eps
            
            S[i, j] = (cost_fn(s0 + ds, a0 + da) - cost_fn(s0 + ds, a0) - 
                      cost_fn(s0, a0 + da) + c0) / (eps * eps)
    
    return Q, R, S, c0

# ----------------------
# Differential Dynamic Programming (DDP)
# ----------------------

@dataclass
class DDPResult:
    """Result of DDP optimization."""
    trajectory: np.ndarray
    controls: np.ndarray
    costs: List[float]
    converged: bool

def ddp(F: Callable, cost_fn: Callable, s_traj: np.ndarray, a_traj: np.ndarray, 
        T: int, max_iter: int = 50, reg: float = 1e-3) -> DDPResult:
    """
    Differential Dynamic Programming (DDP) for trajectory optimization.
    
    DDP is an iterative algorithm that:
    1. Linearizes dynamics and cost around current trajectory
    2. Solves LQR problem for the linearized system
    3. Updates trajectory using the LQR solution
    4. Repeats until convergence
    
    Args:
        F: Dynamics function F(s, a) -> s_next
        cost_fn: Cost function C(s, a) -> cost
        s_traj: Initial state trajectory (T+1, n)
        a_traj: Initial control trajectory (T, m)
        T: Time horizon
        max_iter: Maximum iterations
        reg: Regularization parameter
        
    Returns:
        DDPResult with optimized trajectory and convergence info
    """
    n = s_traj.shape[1]  # State dimension
    m = a_traj.shape[1]  # Control dimension
    
    costs = []
    
    for iteration in range(max_iter):
        # Compute total cost
        total_cost = sum(cost_fn(s_traj[t], a_traj[t]) for t in range(T))
        costs.append(total_cost)
        
        # Linearize around current trajectory
        A_list = []
        B_list = []
        Q_list = []
        R_list = []
        S_list = []
        
        for t in range(T):
            # Linearize dynamics
            A, B, _ = linearize_dynamics(F, s_traj[t], a_traj[t])
            A_list.append(A)
            B_list.append(B)
            
            # Linearize cost
            Q, R, S, _ = linearize_cost_function(cost_fn, s_traj[t], a_traj[t])
            Q_list.append(Q)
            R_list.append(R)
            S_list.append(S)
        
        # Solve LQR for linearized system
        L, P = discrete_lqr(A_list[0], B_list[0], Q_list, R_list, T)
        
        # Update trajectory
        s_new = s_traj.copy()
        a_new = a_traj.copy()
        
        for t in range(T):
            # Compute control update
            du = -L[t] @ (s_new[t] - s_traj[t])
            a_new[t] = a_traj[t] + du
            
            # Simulate forward
            s_new[t+1] = F(s_new[t], a_new[t])
        
        # Check convergence
        if np.linalg.norm(s_new - s_traj) < 1e-6 and np.linalg.norm(a_new - a_traj) < 1e-6:
            print(f"DDP converged after {iteration + 1} iterations")
            break
        
        # Update trajectory
        s_traj = s_new
        a_traj = a_new
    
    return DDPResult(
        trajectory=s_traj,
        controls=a_traj,
        costs=costs,
        converged=iteration < max_iter - 1
    )

# ----------------------
# Linear Quadratic Gaussian (LQG) and Kalman Filter
# ----------------------

class KalmanFilter:
    """
    Kalman Filter for state estimation.
    
    The Kalman filter provides optimal state estimation for linear systems
    with Gaussian noise:
    
    State equation: x_{t+1} = A x_t + B u_t + w_t, w_t ~ N(0, Q)
    Observation equation: y_t = C x_t + v_t, v_t ~ N(0, R)
    
    The filter maintains:
    - State estimate: μ_t = E[x_t | y_0, ..., y_t]
    - Estimation covariance: Σ_t = Cov[x_t | y_0, ..., y_t]
    """
    
    def __init__(self, A: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 init_mean: np.ndarray, init_cov: np.ndarray):
        """
        Initialize Kalman filter.
        
        Args:
            A: State transition matrix
            C: Observation matrix
            Q: Process noise covariance
            R: Observation noise covariance
            init_mean: Initial state estimate
            init_cov: Initial state covariance
        """
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.mean = init_mean.copy()
        self.cov = init_cov.copy()
    
    def predict(self, u: Optional[np.ndarray] = None):
        """
        Prediction step: μ_{t+1}^- = A μ_t + B u_t
                        Σ_{t+1}^- = A Σ_t A^T + Q
        """
        if u is not None:
            # If control input is provided, assume B is identity
            self.mean = self.A @ self.mean + u
        else:
            self.mean = self.A @ self.mean
        
        self.cov = self.A @ self.cov @ self.A.T + self.Q
    
    def update(self, y: np.ndarray):
        """
        Update step: K_t = Σ_t^- C^T (C Σ_t^- C^T + R)^(-1)
                    μ_t = μ_t^- + K_t (y_t - C μ_t^-)
                    Σ_t = (I - K_t C) Σ_t^-
        """
        # Compute Kalman gain
        S = self.C @ self.cov @ self.C.T + self.R
        K = self.cov @ self.C.T @ np.linalg.inv(S)
        
        # Update state estimate
        y_pred = self.C @ self.mean
        self.mean = self.mean + K @ (y - y_pred)
        
        # Update covariance
        I = np.eye(self.cov.shape[0])
        self.cov = (I - K @ self.C) @ self.cov
        
        return self.mean, self.cov

class LQGController:
    """
    Linear Quadratic Gaussian (LQG) controller.
    
    LQG combines LQR control with Kalman filtering:
    1. Use Kalman filter to estimate state from noisy observations
    2. Apply LQR control using the state estimate
    3. Separation principle: optimal control and estimation can be designed separately
    """
    
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, Q_noise: np.ndarray, R_noise: np.ndarray,
                 init_mean: np.ndarray, init_cov: np.ndarray):
        """
        Initialize LQG controller.
        
        Args:
            A, B, C: System matrices
            Q, R: LQR cost matrices
            Q_noise, R_noise: Noise covariance matrices
            init_mean, init_cov: Initial state estimate
        """
        # Design LQR controller
        self.L, _ = discrete_lqr(A, B, Q, R, T=1)  # Infinite horizon approximation
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(A, C, Q_noise, R_noise, init_mean, init_cov)
        
        self.A = A
        self.B = B
        self.C = C
    
    def get_control(self, y: np.ndarray) -> np.ndarray:
        """
        Get control input given observation.
        
        Args:
            y: Current observation
            
        Returns:
            u: Control input
        """
        # Update state estimate
        self.kf.update(y)
        
        # Apply LQR control
        u = -self.L[0] @ self.kf.mean
        
        # Predict for next step
        self.kf.predict(u)
        
        return u

# ----------------------
# Example Usage and Demonstrations
# ----------------------

def demonstrate_finite_horizon_mdp():
    """Demonstrate finite-horizon MDP solution."""
    print("=" * 60)
    print("FINITE-HORIZON MDP DEMONSTRATION")
    print("=" * 60)
    
    # Simple 2-state, 2-action MDP
    states = [0, 1]
    actions = [0, 1]
    T = 3
    
    def P(t, s, a, s_next):
        """Transition probabilities."""
        if t < T:  # Not at terminal time
            if s == 0:
                return 0.7 if s_next == s else 0.3
            else:
                return 0.8 if s_next == s else 0.2
        else:
            return 1.0 if s_next == s else 0.0
    
    def R(t, s, a):
        """Reward function."""
        if t == T:  # Terminal reward
            return 10 if s == 1 else 0
        else:
            return 1 if a == 1 else 0
    
    # Solve finite-horizon MDP
    V, pi = finite_horizon_value_iteration(states, actions, P, R, T)
    
    print("Finite-Horizon MDP Results:")
    print("-" * 40)
    print("Value Function:")
    for t in range(T+1):
        print(f"  t={t}: V({t},0)={V[t,0]:.3f}, V({t},1)={V[t,1]:.3f}")
    
    print("\nOptimal Policy:")
    for t in range(T):
        print(f"  t={t}: π({t},0)={pi[t,0]}, π({t},1)={pi[t,1]}")

def demonstrate_lqr():
    """Demonstrate LQR control."""
    print("\n" + "=" * 60)
    print("LINEAR QUADRATIC REGULATION (LQR) DEMONSTRATION")
    print("=" * 60)
    
    # Simple 1D system: x_{t+1} = x_t + u_t
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = np.array([[1.0]])  # State cost
    R = np.array([[0.1]])  # Control cost
    T = 10
    
    # Solve finite-horizon LQR
    L, P = discrete_lqr(A, B, Q, R, T)
    
    print("LQR Results:")
    print("-" * 40)
    print("Feedback Gains:")
    for t, L_t in enumerate(L):
        print(f"  L_{t} = {L_t[0,0]:.3f}")
    
    # Simulate system
    x0 = np.array([2.0])
    t, x = simulate_lqr_system(A, B, L[0], x0, T=5, dt=0.1)
    
    print(f"\nSimulation Results:")
    print(f"  Initial state: x(0) = {x0[0]:.3f}")
    print(f"  Final state: x({t[-1]:.1f}) = {x[-1,0]:.3f}")
    
    # Plot results
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(t, x[:, 0], 'b-', linewidth=2, label='State x(t)')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('LQR Control Simulation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping plot")

def demonstrate_linearization():
    """Demonstrate dynamics linearization."""
    print("\n" + "=" * 60)
    print("DYNAMICS LINEARIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Nonlinear pendulum dynamics
    def pendulum_dynamics(s, a):
        """Simple pendulum: θ'' + sin(θ) = u"""
        theta, omega = s
        g = 9.81
        L = 1.0
        
        theta_dot = omega
        omega_dot = -g/L * np.sin(theta) + a
        
        return np.array([theta_dot, omega_dot])
    
    # Linearize around equilibrium point
    s0 = np.array([0.0, 0.0])  # Upright position
    a0 = np.array([0.0])       # No control
    
    A, B, c = linearize_dynamics(pendulum_dynamics, s0, a0)
    
    print("Linearization Results:")
    print("-" * 40)
    print("State Jacobian A:")
    print(A)
    print("\nControl Jacobian B:")
    print(B)
    print("\nConstant term c:")
    print(c)
    
    # Verify linearization
    print("\nLinearization Verification:")
    print("-" * 40)
    ds = np.array([0.1, 0.05])
    da = np.array([0.02])
    
    # Nonlinear dynamics
    s_nonlinear = pendulum_dynamics(s0 + ds, a0 + da)
    
    # Linearized dynamics
    s_linear = A @ (s0 + ds) + B @ (a0 + da) + c
    
    print(f"Nonlinear: {s_nonlinear}")
    print(f"Linearized: {s_linear}")
    print(f"Error: {np.linalg.norm(s_nonlinear - s_linear):.6f}")

def demonstrate_ddp():
    """Demonstrate Differential Dynamic Programming."""
    print("\n" + "=" * 60)
    print("DIFFERENTIAL DYNAMIC PROGRAMMING (DDP) DEMONSTRATION")
    print("=" * 60)
    
    # Simple 1D system
    def dynamics(s, a):
        """x_{t+1} = x_t + a_t"""
        return s + a
    
    def cost(s, a):
        """Cost = x^2 + 0.1*a^2"""
        return s**2 + 0.1 * a**2
    
    # Initial trajectory
    T = 10
    s_traj = np.zeros((T+1, 1))
    a_traj = np.zeros((T, 1))
    
    # Set initial state
    s_traj[0] = 2.0
    
    # Run DDP
    result = ddp(dynamics, cost, s_traj, a_traj, T, max_iter=20)
    
    print("DDP Results:")
    print("-" * 40)
    print(f"Converged: {result.converged}")
    print(f"Final cost: {result.costs[-1]:.6f}")
    print(f"Initial state: {result.trajectory[0,0]:.3f}")
    print(f"Final state: {result.trajectory[-1,0]:.3f}")
    
    # Plot cost convergence
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(result.costs, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Total Cost')
        plt.title('DDP Cost Convergence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping plot")

def demonstrate_lqg():
    """Demonstrate LQG control."""
    print("\n" + "=" * 60)
    print("LINEAR QUADRATIC GAUSSIAN (LQG) DEMONSTRATION")
    print("=" * 60)
    
    # System matrices
    A = np.array([[1.0, 0.1],
                  [0.0, 1.0]])  # Double integrator
    B = np.array([[0.0],
                  [0.1]])
    C = np.array([[1.0, 0.0]])  # Only position is observed
    
    # Cost matrices
    Q = np.array([[1.0, 0.0],
                  [0.0, 0.1]])
    R = np.array([[0.1]])
    
    # Noise matrices
    Q_noise = np.array([[0.01, 0.0],
                        [0.0, 0.01]])
    R_noise = np.array([[0.1]])
    
    # Initialize LQG controller
    init_mean = np.array([0.0, 0.0])
    init_cov = np.array([[1.0, 0.0],
                        [0.0, 1.0]])
    
    controller = LQGController(A, B, C, Q, R, Q_noise, R_noise, init_mean, init_cov)
    
    # Simulate system
    T_sim = 50
    x_true = np.zeros((T_sim, 2))
    x_est = np.zeros((T_sim, 2))
    y_obs = np.zeros(T_sim)
    u_control = np.zeros(T_sim)
    
    # Initial conditions
    x_true[0] = np.array([2.0, 0.0])
    x_est[0] = init_mean
    
    for t in range(T_sim - 1):
        # Generate observation
        y_obs[t] = C @ x_true[t] + np.random.normal(0, np.sqrt(R_noise[0,0]))
        
        # Get control
        u_control[t] = controller.get_control(np.array([y_obs[t]]))
        
        # Update true state
        x_true[t+1] = A @ x_true[t] + B @ u_control[t] + np.random.multivariate_normal([0, 0], Q_noise)
        
        # Update estimate
        x_est[t+1] = controller.kf.mean
    
    print("LQG Simulation Results:")
    print("-" * 40)
    print(f"Initial true state: {x_true[0]}")
    print(f"Final true state: {x_true[-1]}")
    print(f"Final estimated state: {x_est[-1]}")
    print(f"Estimation error: {np.linalg.norm(x_true[-1] - x_est[-1]):.6f}")

if __name__ == "__main__":
    # Run comprehensive demonstrations
    demonstrate_finite_horizon_mdp()
    demonstrate_lqr()
    demonstrate_linearization()
    demonstrate_ddp()
    demonstrate_lqg()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("This demonstration shows:")
    print("1. Finite-horizon MDPs with dynamic programming")
    print("2. LQR optimal control for linear systems")
    print("3. Linearization of nonlinear dynamics")
    print("4. DDP for trajectory optimization")
    print("5. LQG control with state estimation")
    print("\nKey insights:")
    print("- Finite-horizon problems require backward induction")
    print("- LQR provides optimal linear feedback control")
    print("- Linearization enables local analysis of nonlinear systems")
    print("- DDP iteratively improves trajectories")
    print("- LQG combines optimal control with state estimation") 