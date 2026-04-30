import numpy as np
from glmbanditexp.utils.utils import mat_norm, dsigmoid, dprobit, solve_glm_mle_dp_sgd_epsilon

from glmbanditexp.utils.binary_tree import BinaryTreeMechanism
import math

import math

import math

##np.random.seed(1234)

def per_step_budget_auto(eps_total, delta_total, k, tol=1e-12, iters=100, switch_k=10):
    """
    Compute per-step (epsilon0, delta0) for composition with k steps.
    Uses naive composition when k is small, advanced composition otherwise.

    Args:
        eps_total : float
            Target total epsilon
        delta_total : float
            Target total delta
        k : int
            Number of compositions
        tol : float
            Tolerance for bisection (advanced comp)
        iters : int
            Max iterations for bisection (advanced comp)
        switch_k : int
            Threshold: if k <= switch_k use naive, else advanced composition.

    Returns:
        epsilon0, delta0 : floats
            Per-step privacy parameters
    """

    if k <= switch_k:
        # ---- naive composition ----
        epsilon0 = eps_total / k
        delta0 = delta_total / k
        return epsilon0, delta0

    # ---- advanced composition ----
    delta_prime = delta_total / 2.0
    delta0 = delta_total / (2.0 * k)
    A = math.sqrt(2.0 * k * math.log(1.0 / delta_prime))

    def f(e0):
        return A*e0 + k*e0*(math.exp(e0) - 1.0) - eps_total

    # initial interval for epsilon0
    lo, hi = 0.0, max(1.0, eps_total / max(A, 1e-12))

    while f(hi) < 0:
        hi *= 2.0
        if hi > 50:  # cap
            break

    for _ in range(iters):
        mid = 0.5*(lo + hi)
        if f(mid) <= 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    epsilon0 = lo
    return epsilon0, delta0


class Private_RS_GLinUCB:
    def __init__(
        self,
        arm_set,
        kappa,
        kappa_star,
        R,
        S,
        model,
        T,
        zeta,
        epsilon,
        delta,
        rng=None,
        save_privacy_search_log='on_failure',
        matrix_diag_every=0,
    ):
        self.name = 'Private-GLM (This work)'
        self.arms = arm_set
        self.d = self.arms[0].shape[0]
        self.K = len(self.arms)
        self.R = R
        self.S = S
        self.T =T
        self.zeta= zeta
        self.kappa = kappa
        self.kappa_star = kappa_star
        self.model = model    
        self.doubling_constant = 1.0
        self.matrix_psd_factor = 5/(8*(math.sqrt(self.d))) + 0.05 #### factor for checking PSD dominance of H matrices; can be tuned for better practical performance (e.g., reduce false positives at higher dimensions)
        self.epsilon = epsilon
        self.delta = delta
        self.save_privacy_search_log = save_privacy_search_log
        self.matrix_diag_every = int(matrix_diag_every)
        self.lmbda =  2.0 *np.log(4/self.delta)*(np.log(4*self.T/self.zeta))**2/ (self.epsilon*self.kappa_star)  + 0.75 * (1 * self.d * np.log(4*self.T / self.zeta)) / (self.R**2) #self.d * np.log(self.T /self.delta)/self.R**2
        print (f"Lambda: {self.lmbda}", "private rs glinucb")
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        self.tree_mechanism_V = BinaryTreeMechanism(T, epsilon/3, delta/3, d = self.d, C = 1, lam =  self.lmbda, rng = self.rng)
        self.tree_mechanism_H = BinaryTreeMechanism(T, epsilon/3, delta/3, d = self.d, C = 1/kappa_star, lam = self.lmbda, rng = self.rng)
        self.gamma = self.R * self.S* np.sqrt(self.d*np.log(self.T/self.zeta))
        self.beta = 4* np.sqrt(self.d* np.log(self.T/self.zeta))
        self.V = (self.lmbda) * np.eye(self.d)
        self.curr_H = self.lmbda * np.eye(self.d)
        self.prev_H = self.lmbda * np.eye(self.d)
        self.max_logdet_prev_H = self._logdet(self.prev_H)
        self.max_logdet_curr_H = self._logdet(self.curr_H)
        # Numerical safeguards for PSD-dominance checks at higher dimensions.
        self.psd_check_tol = 1e-8
        self.psd_check_jitter_rel = 1e-12
        self.warmup_threshold = 1.0 / (0.01 * self.kappa * self.R**4 * self.S**2 * self.d)
        self.t = 1
        self.theta_hat_w = np.zeros((self.d,))
        self.theta_hat_tau = np.zeros((self.d,))
        self.warm_up_X, self.warm_up_Y = [], []
        self.non_warm_up_X, self.non_warm_up_Y = [], []
        self.warmup_flag = False
        self.e = np.exp(1.0)
        self.regret_arr = []
        self.counter1 = 0
        self.counter2 = 0

        self.lmbda_max = self.lmbda + 1.5 *np.log(4/self.delta)*(np.log(4*self.T/self.zeta))**2/ (self.epsilon*self.kappa_star)
        self.lmbda_min = self.lmbda - 1.5 *np.log(4/self.delta)*(np.log(4*self.T/self.zeta))**2/ (self.epsilon*self.kappa_star)

        self.count1_cutoff = max(1, int(self.d*(self.R**2)*self.kappa*(self.gamma**2)*np.log(self.T)/20000))
        self.count2_cutoff = max(1,int(3*np.log2(self.lmbda_max / self.lmbda_min + self.T/(self.kappa_star**2 *self.lmbda_min *self.d))))

        print (f"Count1 cutoff: {self.count1_cutoff}, Count2 cutoff: {self.count2_cutoff}")
        self.thresh1_cross = False
        self.thresh2_cross = False
        
        # Track lambda_max history for debugging
        self.lambda_max_history = []
        self.lambda_max_trigger_times = []

    def _logdet(self, M):
        """Numerically stable log(det(M)) for PSD/SPD matrices."""
        sign, logabsdet = np.linalg.slogdet(M)
        if sign <= 0:
            return -np.inf
        return logabsdet

    def _matrix_diagnostics(self, M):
        vals = np.linalg.eigvalsh(0.5 * (M + M.T))
        min_eig = float(vals[0])
        max_eig = float(vals[-1])
        sign, logabsdet = np.linalg.slogdet(M)
        cond = np.inf if min_eig <= 0 else max_eig / min_eig
        return sign, float(logabsdet), min_eig, float(cond)

    def _not_psd_leq(self, A, B, factor=2.0):
        """
        Return True iff A is NOT <= factor * B in PSD order.

        Uses the largest generalized eigenvalue of (A, B):
            lambda_max(B^{-1/2} A B^{-1/2}) > factor
        This avoids determinant blowups and is stable for moderate/high d.
        """
        # Common positive scaling keeps generalized eigenvalues unchanged,
        # while reducing overflow/underflow risk for very large/small entries.
        scale = max(
            1.0,
            float(np.linalg.norm(A, ord=np.inf)),
            float(np.linalg.norm(B, ord=np.inf)),
        )
        A_sym = 0.5 * ((A / scale) + (A / scale).T)
        B_sym = 0.5 * ((B / scale) + (B / scale).T)
        I = np.eye(self.d)

        # Relative jitter adapts to matrix magnitude after scaling.
        # Using trace/d is more stable than a fixed absolute jitter.
        mean_diag = max(1.0, float(np.trace(B_sym)) / self.d)
        jitter = self.psd_check_jitter_rel * mean_diag
        A_reg = A_sym + jitter * I
        B_reg = B_sym + jitter * I

        try:
            L = np.linalg.cholesky(B_reg)
            # Build S = B^{-1/2} A B^{-1/2} via triangular solves (no explicit inverse).
            Y = np.linalg.solve(L, A_reg)
            S = np.linalg.solve(L, Y.T).T
            S = 0.5 * (S + S.T)
            lambda_max = float(np.linalg.eigvalsh(S)[-1])
            
            # Track the eigenvalue history
            self.lambda_max_history.append(lambda_max)
            
            # Log trigger events (when condition is met)
            if lambda_max > (factor + self.psd_check_tol):
                self.lambda_max_trigger_times.append((self.t, lambda_max, factor))
        except np.linalg.LinAlgError:
            # If B is not numerically SPD, be conservative and trigger refresh.
            return True

        return lambda_max > (factor + self.psd_check_tol)

    def _maybe_log_matrix_diagnostics(self):
        if self.matrix_diag_every <= 0:
            return
        if self.t % self.matrix_diag_every != 0 and self.t != self.T - 1:
            return

        for name, M in (("curr_H", self.curr_H), ("V", self.V)):
            sign, logdet, min_eig, cond = self._matrix_diagnostics(M)
            print(
                f"[Diag][Private-GLM][t={self.t}] {name}: "
                f"sign={sign}, logdet={logdet:.6f}, min_eig={min_eig:.6e}, cond={cond:.6e}"
            )

    def export_lambda_history(self, filepath):
        """Export lambda_max history and trigger events to JSON."""
        import json
        data = {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'lambda_max_history': self.lambda_max_history,
            'lambda_max_trigger_times': [
                {'time': t, 'lambda_max': lmax, 'factor': f}
                for t, lmax, f in self.lambda_max_trigger_times
            ],
            'num_triggers': len(self.lambda_max_trigger_times),
            'max_lambda': max(self.lambda_max_history) if self.lambda_max_history else 0,
            'final_counter2': self.counter2,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Private-GLM] Exported lambda history to {filepath}")
        print(f"  Max λ_max: {data['max_lambda']:.6f}, Num triggers: {data['num_triggers']}")



    def play_arm(self):
        # Check warm up
        self.warmup_flag = False
        max_x, max_norm, max_ind = None, 0, -1
        V_inv = np.linalg.inv(self.V)
        for i in range(self.K):
            x = self.arms[i]
            mnorm = mat_norm(x, V_inv)**2
            if mnorm > self.warmup_threshold:
                if mnorm > max_norm:
                    max_x = x
                    max_norm = mnorm
                    max_ind = i
                self.warmup_flag = True
        if self.warmup_flag:
            self.tree_mechanism_V.update(np.outer(max_x, max_x) / (1.0 + max_norm))
            self.V = self.tree_mechanism_V.query()
            self.a_t = max_ind
            self.tree_mechanism_H.update(np.zeros((self.d, self.d)))
            self.curr_H = self.tree_mechanism_H.query()
            self.max_logdet_curr_H = max(self.max_logdet_curr_H, self._logdet(self.curr_H))
        
        
        # Non warm-up
        else:
            # Check determinant
            self.tree_mechanism_V.update(np.zeros((self.d, self.d)))
            self.V = self.tree_mechanism_V.query()
            
            # Cache inverses once for this round to avoid repeated O(d^3) work per arm.
            V_inv_non_warm = np.linalg.inv(self.V)
            prev_H_inv = np.linalg.inv(self.prev_H)
            
            if self._not_psd_leq(self.curr_H, self.prev_H, factor=1.0 + self.matrix_psd_factor) and not self.thresh2_cross:
                self.prev_H = self.curr_H
                self.max_logdet_prev_H = max(self.max_logdet_prev_H, self._logdet(self.prev_H))
                # Recompute inverse after updating prev_H
                prev_H_inv = np.linalg.inv(self.prev_H)

                # delta_prime = self.delta / 2
                # delta0 = self.delta / (2 * self.count2_cutoff)

                # # per-step ε from advanced composition
                # epsilon0 = self.epsilon / math.sqrt(2 * self.count2_cutoff * math.log(1 / delta_prime))

                epsilon0, delta0 = per_step_budget_auto(self.epsilon/3, self.delta/3, self.count2_cutoff)
                # Adapt starting epochs based on per-step epsilon to help intermediate epsilon regimes
                # (e.g., ε=2) find better configurations without requiring aggressive shrinkage
                if epsilon0 <= 0.1:
                    start_epochs = 15  # Very tight per-step eps: start low
                elif epsilon0 <= 0.3:
                    start_epochs = 25  # Moderate per-step eps: start medium
                elif epsilon0 <= 1.0:
                    start_epochs = 50
                elif epsilon0 <= 2.0:
                    start_epochs = 75
                else:
                    start_epochs = 100 # Loose per-step eps: use standard
                thth,flag = solve_glm_mle_dp_sgd_epsilon(
                    theta_init=(self.theta_hat_w + self.theta_hat_tau)/2,
                    X=np.array(self.non_warm_up_X),
                    Y=np.array(self.non_warm_up_Y),
                    lmbda=1e-5,
                    model=self.model,
                    clip_norm= 0.2,
                    ##theta_ref=self.theta_hat_w,
                    ##S=self.gamma*math.sqrt(self.kappa),
                    epsilon=epsilon0,
                    delta=delta0,
                    num_epochs=start_epochs, batch_size=2048, lr0=10,
                    rng=self.rng,
                    save_privacy_search_log=self.save_privacy_search_log,
                )
                self.counter2 +=1
                ##print ("Diff", np.linalg.norm(thth_test - thth), "shape", np.array(self.non_warm_up_X).shape, "non-warmup")

                self.theta_hat_tau = thth

                if self.counter2 == self.count2_cutoff:
                    self.thresh2_cross = True
                    print (f"Counter2: {self.counter2} == {self.count2_cutoff}, crossing at", self.t, "epsilon", self.epsilon, "delta", self.delta)
                    ##raise ValueError(f"Counter2: {self.counter2} > {self.count2_cutoff}, stopping")
            # Eliminate arms based on warm-up theta
            max_lcb = -np.inf
            arm_idx = []
            ucb_arr = []
            # Compute LCB and UCB of each arm
            for i in range(self.K):
                lcb_idx = np.dot(self.theta_hat_w, self.arms[i]) \
                        - self.gamma * np.sqrt(self.kappa) * mat_norm(self.arms[i], V_inv_non_warm)
                ucb_idx = np.dot(self.theta_hat_w, self.arms[i]) \
                        + self.gamma * np.sqrt(self.kappa) * mat_norm(self.arms[i], V_inv_non_warm)
                ucb_arr.append(ucb_idx)
                if lcb_idx > max_lcb:
                    max_lcb = lcb_idx
            # Eliminate arms
            for i in range(self.K):
                if ucb_arr[i] >= max_lcb:
                    arm_idx.append(i)
            # Find max UCB arm
            max_ind = -np.inf
            self.a_t = -1
            for i in arm_idx:
                ucb_ind = np.dot((self.theta_hat_w + self.theta_hat_tau)/2, self.arms[i]) \
                        + self.beta * mat_norm(self.arms[i], prev_H_inv)
                if ucb_ind > max_ind:
                    max_ind = ucb_ind
                    self.a_t = i
        return self.a_t
    
    def update(self, reward, regret, next_arms):
        self.regret_arr.append(regret)
        # If this was a warm up round, append (x, y) to warm-up set (and non-warm-up set), re-compute theta_hat_w
        if self.warmup_flag and not self.thresh1_cross:
            self.warm_up_X.append(self.arms[self.a_t])
            self.warm_up_Y.append(reward)
            self.non_warm_up_X.append(self.arms[self.a_t])
            self.non_warm_up_Y.append(reward)

            self.counter1 += 1

            # delta_prime = self.delta / 2
            # delta0 = self.delta / (2 * self.count1_cutoff)

            # # per-step ε from advanced composition
            # epsilon0 = self.epsilon / math.sqrt(2 * self.count1_cutoff * math.log(1 / delta_prime))
            epsilon0, delta0 = per_step_budget_auto(self.epsilon/3, self.delta/3, self.count1_cutoff)
            # Adapt starting epochs based on per-step epsilon (same heuristic as non-warmup path)
            if epsilon0 <= 0.1:
                start_epochs = 15
            elif epsilon0 <= 0.3:
                start_epochs = 25
            elif epsilon0 <= 1.0:
                start_epochs = 50
            elif epsilon0 <= 2.0:
                start_epochs = 75
            else:
                start_epochs = 100

            thth, flag = solve_glm_mle_dp_sgd_epsilon(
                theta_init=(self.theta_hat_w + self.theta_hat_tau)/2,
                X=np.array(self.warm_up_X),
                Y=np.array(self.warm_up_Y),
                lmbda=1e-5,
                model=self.model,
                clip_norm = 0.2,
                epsilon=epsilon0,
                delta=delta0,
                num_epochs=start_epochs, batch_size=2048, lr0=10,
                rng=self.rng,
                save_privacy_search_log=self.save_privacy_search_log,
            )

            

            ##print ("Diff", np.linalg.norm(thth_test - thth), "shape", np.array(self.warm_up_X).shape, "warmup")

            if self.counter1 == self.count1_cutoff:
                self.thresh1_cross = True
                print (f"Counter1: {self.counter1} == {self.count1_cutoff}, crossing at", self.t, "epsilon", self.epsilon, "delta", self.delta)
                ##raise ValueError (f"Counter1: {self.counter1} > {self.count1_cutoff}, stopping")
                ##return -1

            self.theta_hat_w = thth

            # if succ_flag:
            #     self.theta_hat_w = thth # update theta_hat_w if mle solution was successful
            #     self.counter1 += 1
        
        # Else add (x, y) to non warm-up set
        elif not self.warmup_flag:
            self.non_warm_up_X.append(self.arms[self.a_t])
            self.non_warm_up_Y.append(reward)
            if self.model == 'Logistic':
                mudp = dsigmoid(np.dot(self.arms[self.a_t], self.theta_hat_w))
            elif self.model == 'Probit':
                mudp = dprobit(np.dot(self.arms[self.a_t], self.theta_hat_w))
            # Update current H matrix
            self.tree_mechanism_H.update(mudp * np.outer(self.arms[self.a_t], self.arms[self.a_t])/self.e)
            self.curr_H = self.tree_mechanism_H.query()
            self.max_logdet_curr_H = max(self.max_logdet_curr_H, self._logdet(self.curr_H))
            
        
        self.t += 1
        self.arms = next_arms
        self._maybe_log_matrix_diagnostics()

        if self.t == self.T:
            print (f"Max logdet curr H: {self.max_logdet_curr_H}, Max logdet prev H: {self.max_logdet_prev_H}", "epsilon", self.epsilon, "delta", self.delta)
            print (f"Counter1 final: {self.counter1}, Counter2 final: {self.counter2}", "epsilon", self.epsilon, "delta", self.delta)
            

            


            


