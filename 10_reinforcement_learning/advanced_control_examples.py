"""
Python implementations and demonstrations for advanced control theory topics:
- Finite-horizon MDPs (Dynamic Programming)
- Linear Quadratic Regulation (LQR)
- Linearization of Nonlinear Dynamics
- Differential Dynamic Programming (DDP)
- Linear Quadratic Gaussian (LQG) and Kalman Filter

Each section is self-contained and includes example usage and comments.
"""
import numpy as np
from scipy.linalg import solve_discrete_are

# --- 1. Finite-horizon MDPs: Value Iteration (Dynamic Programming) ---
def finite_horizon_value_iteration(states, actions, P, R, T):
    """
    Value iteration for finite-horizon MDPs.
    states: list of state indices
    actions: list of action indices
    P: function P(t, s, a, s_next) -> probability
    R: function R(t, s, a) -> reward
    T: time horizon (int)
    Returns: V, pi (optimal value and policy for each time and state)
    """
    nS = len(states)
    nA = len(actions)
    V = np.zeros((T+1, nS))
    pi = np.zeros((T, nS), dtype=int)
    # Terminal value
    for s in states:
        V[T, s] = max(R(T, s, a) for a in actions)
    # Backward induction
    for t in reversed(range(T)):
        for s in states:
            Q = [R(t, s, a) + sum(P(t, s, a, s_next) * V[t+1, s_next] for s_next in states) for a in actions]
            V[t, s] = max(Q)
            pi[t, s] = np.argmax(Q)
    return V, pi

# Example usage:
if __name__ == "__main__":
    print("--- Finite-horizon MDP Example ---")
    states = [0, 1]
    actions = [0, 1]
    T = 3
    def P(t, s, a, s_next):
        return 0.7 if s_next == s else 0.3
    def R(t, s, a):
        return 1 if a == 1 else 0
    V, pi = finite_horizon_value_iteration(states, actions, P, R, T)
    print("Optimal Value Function:\n", V)
    print("Optimal Policy:\n", pi)

# --- 2. Linear Quadratic Regulation (LQR) ---
def lqr(A, B, Q, R, T):
    """
    Finite-horizon discrete-time LQR via dynamic programming.
    Returns: list of feedback gains L_t, value matrices P_t, and value offsets.
    """
    n = A.shape[0]
    m = B.shape[1]
    P = [None] * (T+1)
    L = [None] * T
    P[T] = Q[T]
    for t in reversed(range(T)):
        BtPB = B.T @ P[t+1] @ B
        BtPA = B.T @ P[t+1] @ A
        L[t] = np.linalg.solve(BtPB + R[t], BtPA).T
        P[t] = Q[t] + A.T @ P[t+1] @ (A - B @ L[t])
    return L, P

# Example usage:
if __name__ == "__main__":
    print("\n--- LQR Example ---")
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = [np.array([[1.0]]) for _ in range(4)]
    R = [np.array([[1.0]]) for _ in range(4)]
    T = 3
    L, P = lqr(A, B, Q, R, T)
    print("Feedback gains L_t:", L)

# --- 3. Linearization of Nonlinear Dynamics ---
def linearize_dynamics(F, s0, a0):
    """
    Linearize a nonlinear function F(s, a) around (s0, a0) using finite differences.
    Returns: A, B, c such that F(s, a) ≈ A(s-s0) + B(a-a0) + c
    """
    s0 = np.atleast_1d(s0)
    a0 = np.atleast_1d(a0)
    n = s0.size
    m = a0.size
    eps = 1e-5
    f0 = F(s0, a0)
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    # Jacobian w.r.t. s
    for i in range(n):
        ds = np.zeros(n)
        ds[i] = eps
        A[:, i] = (F(s0 + ds, a0) - f0) / eps
    # Jacobian w.r.t. a
    for j in range(m):
        da = np.zeros(m)
        da[j] = eps
        B[:, j] = (F(s0, a0 + da) - f0) / eps
    c = f0 - A @ s0 - B @ a0
    return A, B, c

# Example usage:
if __name__ == "__main__":
    print("\n--- Linearization Example ---")
    def F(s, a):
        return np.array([np.sin(s[0]) + a[0]])
    s0 = np.array([0.0])
    a0 = np.array([1.0])
    A, B, c = linearize_dynamics(F, s0, a0)
    print("A:", A, "B:", B, "c:", c)

# --- 4. Differential Dynamic Programming (DDP) ---
def ddp(F, R, s_traj, a_traj, T):
    """
    Illustrative DDP: linearize around nominal trajectory, then solve LQR.
    F: function F(s, a) -> s_next
    R: function R(s, a) -> reward
    s_traj, a_traj: nominal trajectories (arrays)
    T: time horizon
    Returns: L_t (feedback gains)
    """
    n = s_traj.shape[1]
    m = a_traj.shape[1]
    A_list, B_list, Q_list, R_list = [], [], [], []
    for t in range(T):
        A, B, _ = linearize_dynamics(F, s_traj[t], a_traj[t])
        A_list.append(A)
        B_list.append(B)
        # Quadratic cost approximation (here, just identity for demo)
        Q_list.append(np.eye(n))
        R_list.append(np.eye(m))
    L, _ = lqr(A_list[0], B_list[0], Q_list, R_list, T)
    return L

# Example usage:
if __name__ == "__main__":
    print("\n--- DDP Example ---")
    T = 3
    s_traj = np.zeros((T+1, 1))
    a_traj = np.zeros((T, 1))
    def F(s, a):
        return np.array([s[0] + a[0]])
    def R(s, a):
        return -s[0]**2 - a[0]**2
    L = ddp(F, R, s_traj, a_traj, T)
    print("DDP feedback gains:", L)

# --- 5. Linear Quadratic Gaussian (LQG) and Kalman Filter ---
class KalmanFilter:
    def __init__(self, A, C, Q, R, init_mean, init_cov):
        self.A = A
        self.C = C
        self.Q = Q  # process noise cov
        self.R = R  # observation noise cov
        self.mean = init_mean
        self.cov = init_cov
    def predict(self):
        self.mean = self.A @ self.mean
        self.cov = self.A @ self.cov @ self.A.T + self.Q
    def update(self, y):
        S = self.C @ self.cov @ self.C.T + self.R
        K = self.cov @ self.C.T @ np.linalg.inv(S)
        self.mean = self.mean + K @ (y - self.C @ self.mean)
        self.cov = self.cov - K @ self.C @ self.cov
        return self.mean, self.cov

# Example usage:
if __name__ == "__main__":
    print("\n--- Kalman Filter Example (LQG) ---")
    A = np.array([[1.0]])
    C = np.array([[1.0]])
    Q = np.array([[0.01]])
    R = np.array([[0.1]])
    init_mean = np.array([0.0])
    init_cov = np.array([[1.0]])
    kf = KalmanFilter(A, C, Q, R, init_mean, init_cov)
    true_state = 0.0
    for t in range(5):
        kf.predict()
        obs = true_state + np.random.normal(0, np.sqrt(R[0,0]))
        mean, cov = kf.update(np.array([obs]))
        print(f"Step {t}: Observation={obs:.2f}, Estimate={mean[0]:.2f}, Cov={cov[0,0]:.2f}") 