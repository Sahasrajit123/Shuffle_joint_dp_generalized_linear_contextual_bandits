
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from scipy.optimize import brentq
import logging
import os
from pathlib import Path

##np.random.seed(42)

# Set up default logger for DP-SGD messages
_loggers_cache = {}  # Cache loggers by run_config

def _get_default_logger(run_config=None):
    """
    Get or create the default logger for DP-SGD messages.
    
    Args:
        run_config: String identifier for the run configuration. 
                   If None, uses 'default'. Creates subfolder logs/{run_config}/
    
    Returns:
        logging.Logger instance
    """
    # Use default if not provided
    if run_config is None:
        run_config = 'default'
    
    # Return cached logger if it exists
    if run_config in _loggers_cache:
        return _loggers_cache[run_config]
    
    # Create logs directory structure
    logs_base_dir = Path('logs')
    logs_base_dir.mkdir(exist_ok=True)
    
    # Create run config subdirectory
    run_config_dir = logs_base_dir / run_config
    run_config_dir.mkdir(exist_ok=True)
    
    # Create log file path
    log_file = run_config_dir / 'dp_sgd.log'
    
    # Create logger with unique name per run_config
    logger_name = f'dp_sgd_{run_config}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Cache the logger
    _loggers_cache[run_config] = logger
    
    return logger

def adapt_training_parameters_for_noise(noise_scale, num_epochs, burn_in_epochs, lr0, 
                                       epsilon, delta, n, batch_size, clip_norm,
                                       verbose=False, logger=None, use_default_logger=True,
                                       run_config=None):
    """
    Iteratively adapt training parameters when noise_scale is large.
    
    Args:
        noise_scale: Current noise scale
        num_epochs: Original number of epochs
        burn_in_epochs: Original burn-in epochs  
        lr0: Original learning rate
        epsilon, delta: Privacy parameters
        n: Dataset size
        batch_size: Batch size
        clip_norm: Gradient clipping norm
        verbose: If True, print adaptation messages to stdout
        logger: Optional logging.Logger instance to write messages to log file
        use_default_logger: If True and logger is None, use default logger to write to logs/{run_config}/dp_sgd.log
        run_config: String identifier for the run configuration (creates subfolder logs/{run_config}/)
        
    Returns:
        tuple: (adapted_epochs, adapted_burn_in, adapted_lr, final_steps, final_noise_scale)
    """
    # Auto-generate run_config from epsilon/delta if not provided
    if run_config is None:
        # Create a meaningful run_config identifier from epsilon and delta
        eps_str = f"eps{epsilon:.2f}".replace('.', '_')
        delta_str = f"delta{delta:.2e}".replace('.', '_').replace('e-', 'e')
        run_config = f"{eps_str}_{delta_str}"
    
    # Use default logger if no logger provided and use_default_logger is True
    if logger is None and use_default_logger:
        logger = _get_default_logger(run_config=run_config)
    
    def _log(msg):
        """Helper to log/print messages based on verbose and logger settings."""
        if logger:
            logger.info(msg)
        if verbose:
            print(msg)
    
    if noise_scale <= 0.3:
        # No adaptation needed - noise scale is manageable
        steps = num_epochs * int(np.ceil(n / batch_size))
        return num_epochs, burn_in_epochs, lr0, steps, noise_scale
    
    _log(f"Large noise scale detected ({noise_scale:.4f}). Starting iterative adaptation...")
    
    # Iterative parameter adjustment
    max_iterations = 10
    min_epochs = 15  # Increased minimum epochs for stability
    min_batches_per_epoch = 2
    target_noise_scale = 0.2  # Target noise scale for good convergence
    
    current_epochs = num_epochs
    current_burn_in = burn_in_epochs
    current_lr = lr0
    
    for iteration in range(max_iterations):
        # Calculate current steps and noise scale
        current_steps = current_epochs * int(np.ceil(n / batch_size))
        current_q = batch_size / n
        current_sigma = compute_noise_multiplier(epsilon, delta, current_steps, current_q)
        current_noise_scale = current_sigma * clip_norm / batch_size
        
        _log(f"  Iteration {iteration + 1}: epochs={current_epochs}, steps={current_steps}, noise_scale={current_noise_scale:.4f}")
        
        # Check if we've reached a good balance
        if current_noise_scale <= target_noise_scale:
            _log(f"  ✓ Achieved target noise scale ({current_noise_scale:.4f} <= {target_noise_scale})")
            break
            
        # Check minimum constraints
        if current_epochs <= min_epochs:
            _log(f"  ⚠ Reached minimum epochs ({min_epochs}), stopping adaptation")
            break
            
        # Check if we have enough total batches for meaningful training
        total_batches = current_epochs * int(np.ceil(n / batch_size))
        min_total_batches = 20  # Minimum total batches needed for stable training
        
        if total_batches < min_total_batches:
            _log(f"  ⚠ Too few total batches ({total_batches} < {min_total_batches}), stopping adaptation")
            break
        
        # Adaptive reduction: more conservative to prevent extreme reduction
        if current_noise_scale > 2.0:
            reduction_factor = 0.8  # Conservative for extremely high noise
        elif current_noise_scale > 1.0:
            reduction_factor = 0.85  # Moderate for very high noise
        elif current_noise_scale > 0.5:
            reduction_factor = 0.9  # Gentle for high noise
        else:
            reduction_factor = 0.95  # Very gentle for moderate noise
            
        # Apply reduction
        new_epochs = max(min_epochs, int(current_epochs * reduction_factor))
        new_burn_in = max(1, int(current_burn_in * reduction_factor))
        
        # Only update if we're making progress
        if new_epochs < current_epochs:
            current_epochs = new_epochs
            current_burn_in = new_burn_in
            # Adjust learning rate: normalize if too high, increase if reasonable
            if current_lr > 2.0:  # If LR is too high, normalize it
                current_lr = min(1.0, current_lr * 0.8)  # Reduce towards reasonable range
            elif current_lr < 1.0:  # If LR is reasonable, can increase slightly
                current_lr = min(current_lr * 1.1, 1.5)  # Small increase, capped at 1.5
        else:
            _log(f"  ⚠ No further reduction possible, stopping adaptation")
            break
    
    # Final calculations
    final_steps = current_epochs * int(np.ceil(n / batch_size))
    final_sigma = compute_noise_multiplier(epsilon, delta, final_steps, batch_size / n)
    final_noise_scale = final_sigma * clip_norm / batch_size
    
    _log(f"Final adaptive parameters:")
    _log(f"  - Epochs: {num_epochs} → {current_epochs}")
    _log(f"  - Burn-in: {burn_in_epochs} → {current_burn_in}")
    _log(f"  - Learning rate: {lr0:.3f} → {current_lr:.3f}")
    _log(f"  - Steps: {num_epochs * int(np.ceil(n / batch_size))} → {final_steps}")
    _log(f"  - Final noise scale: {noise_scale:.4f} → {final_noise_scale:.4f}")
    
    return current_epochs, current_burn_in, current_lr, final_steps, final_noise_scale


def weighted_norm(x, A):
    return np.sqrt(np.dot(x, np.dot(A, x)))


def gaussian_sample_ellipsoid(center, design, radius, rng=None):
    dim = len(center)
    if rng is not None:
        sample = rng.normal(0, 1, (dim,))
    else:
        sample = np.random.normal(0, 1, (dim,))
    res = np.real_if_close(center + np.linalg.solve(sqrtm(design), sample) * radius)
    return res


def mat_norm(vec, matrix):
    return np.sqrt(np.dot(vec, np.dot(matrix, vec)))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def log_loss_glm(theta, X, Y, lmbda, model):
    if X.shape[0] == 0:
        raise ValueError("Cannot compute loss with empty data")
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    
    if model == 'Logistic':
        return - np.sum(Y * np.log(sigmoid(np.dot(X, theta))) + (1 - Y) * np.log(1 - sigmoid(np.dot(X, theta)))) + lmbda * np.sum(np.square(theta))
    elif model == 'Probit':
        return - np.sum(Y * np.log(probit(np.dot(X, theta))) + (1 - Y) * np.log(1 - probit(np.dot(X, theta)))) + lmbda * np.sum(np.square(theta))

def grad_log_loss_glm(theta, X, Y, lmbda, model):
    """
    Gradient of regularised NLL for mini‑batch X (b×d), Y (b,) or (b,1).
    """
    if X.shape[0] == 0:
        raise ValueError("Cannot compute gradient with empty data")
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    
    Y = Y.ravel()                    # ensure 1‑D
    if model == 'Logistic':
        p = sigmoid(X @ theta)
    elif model == 'Probit':
        p = probit(X @ theta)
    else:
        raise ValueError("model must be 'Logistic' or 'Probit'")
    return X.T @ (p - Y) + 2.0 * lmbda * theta

def grad_log_loss_glm_sgd(theta, X, Y, model):
    """
    Gradient of regularised NLL for single sample X (d,) and Y (scalar).
    """
    if model == 'Logistic':
        p = sigmoid(np.dot(X, theta))
        return X.T @ (p - Y)/X.shape[0]
    elif model == 'Probit':
        p = probit(np.dot(X, theta))
        return X.T @ (p - Y)/X.shape[0]
    else:
        raise ValueError("model must be 'Logistic' or 'Probit'")


def hess_log_loss_glm(theta, X, Y, lmbda, model):
    if model == 'Logistic':
        return np.sum([dsigmoid(np.dot(theta, x)) * np.outer(x, x) for x in X], axis=0) + lmbda*np.eye(theta.shape[0])
    elif model == 'Probit':
        return np.sum([dprobit(np.dot(theta, x)) * np.outer(x, x) for x in X], axis=0) + lmbda*np.eye(theta.shape[0])

def probit(x):
    return norm.cdf(x)

def dprobit(x):
    return (1.0 / np.sqrt(2*np.pi)) * np.exp(-x*x/2.0)

def solve_glm_mle(theta_prev, X, Y, lmbda, model):
    # Check if we have any data
    if X.shape[0] == 0:
        # No data available, return previous theta
        return theta_prev, True
    
    # Additional check: ensure X and Y have compatible shapes
    if X.shape[0] != Y.shape[0]:
        return theta_prev, True
    
    # res = minimize(log_loss_glm, theta_prev,\
    #                jac=grad_log_loss_glm, hess=hess_log_loss_glm, \
    #                 args=(X, Y, lmbda, model), method='Newton-CG')
    res = minimize(log_loss_glm, theta_prev, args=(X, Y, lmbda, model), jac=grad_log_loss_glm, method='L-BFGS-B')
    # if not res.success:
    #     print(res.message)
    theta_hat, succ_flag = res.x, res.success
    return theta_hat, succ_flag

def project_onto_ball(theta, theta_ref, S):
    diff = theta - theta_ref
    norm = np.linalg.norm(diff)
    if norm > S:
        return theta_ref + (S / norm) * diff
    return theta

def compute_noise_multiplier(epsilon, delta, steps, sample_rate, sigma_bounds = (1e-5, 4e7), tol=1e-6): 
    def get_rdp_orders(target_epsilon):
        if target_epsilon >= 0.3:
            return np.arange(2.0, 64.0, 1.0)
        else:
            return np.concatenate([np.arange(2.0, 64.0, 4.0), np.arange(70, 512, 6)])
        
    def rdp_gaussian(q, sigma, steps, alpha):
        if q == 0:
            return 0
        return steps * (alpha * q**2) / (2 * sigma**2)
    
    orders = get_rdp_orders(epsilon)

    def compute_epsilon(sigma):
        epsilons = []
        for alpha in orders:
            rdp = rdp_gaussian(sample_rate, sigma, steps, alpha)
            eps = rdp - np.log(delta) / (alpha - 1)   # Correct RDP to ε conversion
            epsilons.append(eps)
        return min(epsilons)

    # Binary search to find sigma such that ε <= target
    ##sigma_min, sigma_max = 1e-3, 4e8

    def objective(sigma):
        return compute_epsilon(sigma) - epsilon

    # Check that the bounds are valid
    eps_low = compute_epsilon(sigma_bounds[0])
    eps_high = compute_epsilon(sigma_bounds[1])



    if eps_low < epsilon and eps_high < epsilon:
        raise ValueError ("q", sample_rate, "steps", steps, "orders", orders, "sigma_bounds", sigma_bounds, "tol", tol, "ε target is too high: even the smallest σ is over-private.", eps_low, eps_high, epsilon,delta)
        
    if eps_low > epsilon and eps_high > epsilon:
        raise ValueError ("q", sample_rate, "steps", steps, "orders", orders, "sigma_bounds", sigma_bounds, "tol", tol, "ε target is too tight: even the largest σ doesn't meet ε.", eps_low, eps_high, epsilon,delta)

    # Root-finding for objective(sigma) = 0
    sigma_opt = brentq(objective, *sigma_bounds, xtol=tol)
    return sigma_opt   
       
    ##return sigma_max

# def solve_glm_mle_dp_sgd_epsilon(theta_init, X, Y, lmbda, model,
#                               theta_ref, S,
#                               epsilon, delta,
#                               num_epochs=20, batch_size=64,
#                               lr=1e-2, clip_norm=1.0, seed=42):
#     np.random.seed(seed)
#     n, d = X.shape
#     batch_size = min(batch_size, n)              # ensure batch size doesn't exceed dataset
#     if n <= 30:
#             num_epochs = min(num_epochs, 4)         # reduce epochs if dataset is very small


#     steps = max(1, num_epochs * (n // batch_size))
#     sample_rate = batch_size / n

#     steps = num_epochs * (n // batch_size)
#     sample_rate = batch_size / n
#     noise_multiplier = compute_noise_multiplier(epsilon, delta, steps, sample_rate)

#     theta = theta_init.copy()
#     for epoch in range(num_epochs):
#         perm = np.random.permutation(n)
#         X_shuffled, Y_shuffled = X[perm], Y[perm]

#         for i in range(0, n, batch_size):
#             X_batch = X_shuffled[i:i+batch_size]
#             Y_batch = Y_shuffled[i:i+batch_size]
#             b = X_batch.shape[0]

#             # Per-sample gradients using existing grad_log_loss_glm
#             grads = []
#             for j in range(b):
#                 grad_j = grad_log_loss_glm(theta, X_batch[j:j+1], Y_batch[j:j+1], lmbda, model)
#                 norm_j = np.linalg.norm(grad_j)
#                 if norm_j > clip_norm:
#                     grad_j = grad_j * (clip_norm / norm_j)
#                 grads.append(grad_j)
#             grads = np.array(grads)

#             # Noisy gradient
#             grad_mean = np.mean(grads, axis=0)
#             noise = np.random.normal(0, noise_multiplier * clip_norm / b, size=theta.shape)
#             noisy_grad = grad_mean + noise

#             # Gradient step and projection
#             theta = theta - lr * noisy_grad
#             theta = project_onto_ball(theta, theta_ref, S)

#     return theta, True

# def solve_glm_mle_sgd(theta_init, X, Y, lmbda, model,
#                       theta_ref, S,
#                       num_epochs=20, batch_size=64,
#                       lr=1e-2, clip_norm=None, seed=42):
#     """
#     Non-private projected SGD for fitting GLM (Logistic/Probit).
#     """
#     np.random.seed(seed)
#     n, d = X.shape
#     batch_size = n ##min(batch_size, n)  # avoid batch_size > n

#     # if n <= 30:
#     #     num_epochs = min(num_epochs, 4)

#     theta = theta_init.copy()
#     for epoch in range(num_epochs):
#         perm = np.random.permutation(n)
#         X_shuffled, Y_shuffled = X[perm], Y[perm]

#         for i in range(0, n, batch_size):
#             X_batch = X_shuffled[i:i+batch_size]
#             Y_batch = Y_shuffled[i:i+batch_size]
#             b = X_batch.shape[0]

#             grad = grad_log_loss_glm(theta, X_batch, Y_batch, lmbda, model)

#             if clip_norm is not None:
#                 norm = np.linalg.norm(grad)
#                 if norm > clip_norm:
#                     grad *= (clip_norm / norm)

#             theta = theta - lr * grad
#             ##theta = project_onto_ball(theta, theta_ref, S)

#     return theta, True

# ------------------------------------------------------------------
# SGD that matches SciPy's optimum (Logistic / Probit)
# ------------------------------------------------------------------
def solve_glm_mle_sgd(
        theta_init, X, Y, lmbda, model,
        num_epochs = 30,
        batch_size = 1024,
        lr0        = 0.5,
        momentum   = 0.0,     # 0.0 → plain SGD; still DP‑compatible
        clip_norm  = None,
        seed       = 0,
        tol_theta  = 1e-6,
        tol_obj    = 1e-9,
        burn_in_epochs = 10
):
    """
    Optimises the *sum* objective
        Σ_i loss_i(θ) + λ‖θ‖²
    with SGD + (optional) momentum and   η_t = lr0 / √t.
    Everything here can be used inside DP‑SGD (just add noise after
    clipping; σ does not rely on gradient history).
    """
    rng   = np.random.default_rng(seed)
    n, d  = X.shape
    theta = theta_init.astype(float, copy=True)
    v     = np.zeros_like(theta)          # momentum buffer
    step  = 0

    obj = lambda th: log_loss_glm(th, X, Y, lmbda, model)
    prev_obj = obj(theta)

    idx_all = np.arange(n)

    theta_avg = np.zeros_like(theta, dtype=float)
    k = 0

    batch_size = min(batch_size, n)  # ensure batch size doesn't exceed dataset

    for epoch in range(num_epochs):
        rng.shuffle(idx_all)

        for start in range(0, n, batch_size):
            step += 1
            lr_t = lr0 / np.sqrt(step)    # √t schedule

            idx = idx_all[start:start + batch_size]
            g   = grad_log_loss_glm_sgd(theta, X[idx], Y[idx],
                                    lmbda, model) 

            # optional clipping
            if clip_norm is not None:
                g_norm = np.linalg.norm(g)
                if g_norm > clip_norm:
                    g *= clip_norm / g_norm

            # momentum update (classical, DP‑safe)

            g += 2.0 * lmbda * theta  # add regularization gradient
            v = momentum * v + g
            theta -= lr_t * v if momentum > 0.0 else lr_t * g

        # early‑stop once per epoch
        cur_obj = obj(theta)
        if np.linalg.norm(lr_t * v) < tol_theta and abs(cur_obj - prev_obj) < tol_obj:
            break
        prev_obj = cur_obj

        if epoch >= burn_in_epochs:
            theta_avg += theta
            k += 1

    return theta_avg/k, True

def solve_glm_mle_dp_sgd_epsilon(
        theta_init, X, Y, lmbda, model,
        epsilon, delta,
        num_epochs   = 30,
        batch_size   = 1024,
        lr0          = 0.5,
        clip_norm    = 1.0,
        tail_batches = 100,      # how many final updates to average
        theta_ref    = None,
        S            = None,
        burn_in_epochs = 10,
        rng          = None,
        verbose      = False,
        logger       = None,
        use_default_logger = True,
        run_config   = None
):
    """
    DP‑SGD that returns the **average** of the last `tail_batches` parameter
    vectors (post‑processing → same (ε,δ) guarantee).
    
    Args:
        verbose: If True, print DP-SGD messages to stdout
        logger: Optional logging.Logger instance to write messages to log file
        use_default_logger: If True and logger is None, use default logger to write to logs/{run_config}/dp_sgd.log
        run_config: String identifier for the run configuration (creates subfolder logs/{run_config}/)
    """
    # Auto-generate run_config from epsilon/delta if not provided
    if run_config is None:
        # Create a meaningful run_config identifier from epsilon and delta
        eps_str = f"eps{epsilon:.2f}".replace('.', '_')
        delta_str = f"delta{delta:.2e}".replace('.', '_').replace('e-', 'e')
        run_config = f"{eps_str}_{delta_str}"
    
    # Use default logger if no logger provided and use_default_logger is True
    if logger is None and use_default_logger:
        logger = _get_default_logger(run_config=run_config)
    
    def _log(msg):
        """Helper to log/print messages based on verbose and logger settings."""
        if logger:
            logger.info(msg)
        if verbose:
            print(msg)
    
    n, d = X.shape
    theta = theta_init.astype(float, copy=True)
    batch_size = min(batch_size, n)  # ensure batch size doesn't exceed dataset

    # — noise multiplier σ chosen via RDP —
    steps       = num_epochs * int(np.ceil(n / batch_size))
    q           = batch_size / n
    
    # Try to compute sigma, with fallback adaptation if it fails
    try:
        sigma = compute_noise_multiplier(epsilon, delta, steps, q)
        noise_scale = sigma * clip_norm / batch_size
        _log(f"DP-SGD: σ = {sigma:.4f} for ε={epsilon}, δ={delta}, steps={steps}, q={q} noise scale {noise_scale}")
        
        # Adaptive parameter adjustment for large noise scales
        num_epochs, burn_in_epochs, lr0, steps, noise_scale = adapt_training_parameters_for_noise(
            noise_scale, num_epochs, burn_in_epochs, lr0, epsilon, delta, n, batch_size, clip_norm,
            verbose=verbose, logger=logger, use_default_logger=use_default_logger, run_config=run_config
        )
        
    except ValueError as e:
        error_msg = str(e)
        # Only handle specific privacy parameter errors, not other ValueErrors
        if "ε target is too tight" in error_msg or "ε target is too high" in error_msg:
            _log(f"⚠ Privacy parameters too strict: {error_msg}")
            _log("Attempting adaptation to reduce steps and recompute...")
            
            # Try adaptation with progressively fewer epochs
            original_epochs = num_epochs
            adaptation_success = False
            
            for reduction_factor in [0.8, 0.6, 0.4, 0.2]:
                try:
                    adapted_epochs = max(5, int(original_epochs * reduction_factor))
                    adapted_steps = adapted_epochs * int(np.ceil(n / batch_size))
                    
                    _log(f"  Trying {adapted_epochs} epochs ({adapted_steps} steps)...")
                    sigma = compute_noise_multiplier(epsilon, delta, adapted_steps, q)
                    noise_scale = sigma * clip_norm / batch_size
                    
                    _log(f"✓ Success! σ = {sigma:.4f}, noise_scale = {noise_scale:.4f}")
                    
                    # Update parameters
                    num_epochs = adapted_epochs
                    steps = adapted_steps
                    burn_in_epochs = max(1, int(burn_in_epochs * reduction_factor))
                    
                    # Apply additional adaptation if noise scale is still high
                    if noise_scale > 0.3:
                        num_epochs, burn_in_epochs, lr0, steps, noise_scale = adapt_training_parameters_for_noise(
                            noise_scale, num_epochs, burn_in_epochs, lr0, epsilon, delta, n, batch_size, clip_norm,
                            verbose=verbose, logger=logger, use_default_logger=use_default_logger, run_config=run_config
                        )
                    
                    adaptation_success = True
                    break
                    
                except ValueError as inner_e:
                    inner_error_msg = str(inner_e)
                    if "ε target is too tight" in inner_error_msg or "ε target is too high" in inner_error_msg:
                        _log(f"  Failed with {adapted_epochs} epochs: {inner_error_msg}")
                        continue
                    else:
                        # Re-raise if it's a different ValueError
                        raise inner_e
            
            # If no adaptation succeeded, raise error
            if not adaptation_success:
                raise ValueError(f"Could not compute sigma even after adaptation. Original error: {e}")
        else:
            # If we reach here, it's not a privacy parameter error, so re-raise
            raise e
    # storage for tail‑averaging
    tail_buf = []
    max_tail = tail_batches

    t_step = 0
    idx_all = np.arange(n)
    theta_avg = np.zeros_like(theta, dtype=float)
    k=0

    for epoch in range(num_epochs):
        np.random.shuffle(idx_all)
        for start in range(0, n, batch_size):
            t_step += 1
            lr_t = lr0 / np.sqrt(t_step)

            idx  = idx_all[start:start + batch_size]
            X_b, Y_b = X[idx], Y[idx]

            # per‑sample gradients + clipping
            grads = []
            for x_i, y_i in zip(X_b, Y_b):
                g_i = grad_log_loss_glm_sgd(
                    theta, x_i[None, :], np.array([y_i]),
                    model
                )
                g_i_norm = np.linalg.norm(g_i)
                if g_i_norm > clip_norm:
                    g_i *= clip_norm / g_i_norm
                grads.append(g_i)
            g_bar = np.mean(grads, axis=0) + 2.0 * lmbda * theta  # add regularization gradient

            # add DP noise
            noise = np.random.normal(0.0, noise_scale, size=d)
            g_priv = g_bar + noise

            # SGD update
            theta -= lr_t * g_priv

            # optional projection
            if theta_ref is not None and S is not None:
                diff = theta - theta_ref
                norm = np.linalg.norm(diff)
                if norm > S:
                    theta = theta_ref + diff * (S / norm)

            # # ---- tail buffer update ----
            # tail_buf.append(theta.copy())
            # if len(tail_buf) > max_tail:
            #     tail_buf.pop(0)
        if epoch >= burn_in_epochs:
            theta_avg += theta
            k += 1
        

    # post‑processing: average of last `tail_batches` thetas
    ##theta_avg = np.mean(tail_buf, axis=0)
    return theta_avg/k, True


