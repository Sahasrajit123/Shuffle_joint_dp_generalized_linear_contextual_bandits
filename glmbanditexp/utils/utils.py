
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from scipy.optimize import brentq
import logging
import os
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4

try:
    from dp_accounting import dp_event, rdp, privacy_accountant
except ImportError:
    dp_event = None
    rdp = None
    privacy_accountant = None

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


def _write_privacy_search_log(run_config, report):
    """
    Persist privacy search attempts in a unique .log file under logs/{run_config}/.
    One solver invocation => one report file.
    """
    if run_config is None:
        run_config = 'default'

    logs_base_dir = Path('logs')
    logs_base_dir.mkdir(exist_ok=True)

    run_config_dir = logs_base_dir / run_config
    run_config_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')
    unique_id = uuid4().hex[:8]
    report_file = run_config_dir / f'dp_privacy_search_{timestamp}_{unique_id}.log'

    attempts = report.get("attempts", [])
    best_attempt = report.get("best_attempt")
    selected = report.get("selected_calibration")

    def _fmt_float(val, digits=6):
        if val is None:
            return "None"
        try:
            return f"{float(val):.{digits}g}"
        except Exception:
            return str(val)

    with open(report_file, 'w') as f:
        f.write("DP Privacy Calibration Search Report\n")
        f.write(f"run_config: {report.get('run_config')}\n")
        f.write(f"epsilon: {report.get('epsilon')}\n")
        f.write(f"delta: {report.get('delta')}\n")
        f.write(f"initial_epochs: {report.get('initial_epochs')}\n")
        f.write(f"initial_batch_size: {report.get('initial_batch_size')}\n")
        f.write(f"attempt_count: {report.get('attempt_count')}\n")
        f.write("\n")

        if selected is not None:
            f.write("Selected calibration:\n")
            f.write(
                "  "
                f"epochs={selected.get('epochs')}, "
                f"batch_size={selected.get('batch_size')}, "
                f"steps={selected.get('steps')}, "
                f"q={_fmt_float(selected.get('q'))}, "
                f"sigma={_fmt_float(selected.get('sigma'))}, "
                f"noise_scale={_fmt_float(selected.get('noise_scale'))}\n"
            )
            f.write("\n")
        else:
            f.write("Selected calibration: None\n\n")

        if best_attempt is not None:
            f.write("Best attempt:\n")
            f.write(
                "  "
                f"feasible={best_attempt.get('feasible')}, "
                f"status={best_attempt.get('status')}, "
                f"epochs={best_attempt.get('epochs')}, "
                f"batch_size={best_attempt.get('batch_size')}, "
                f"steps={best_attempt.get('steps')}, "
                f"q={_fmt_float(best_attempt.get('q'))}, "
                f"sigma={_fmt_float(best_attempt.get('sigma'))}, "
                f"noise_scale={_fmt_float(best_attempt.get('noise_scale'))}, "
                f"eps_high={_fmt_float(best_attempt.get('eps_high'))}, "
                f"epsilon_gap={_fmt_float(best_attempt.get('epsilon_gap'))}\n"
            )
            f.write("\n")

        f.write("Attempts tried (in order):\n")
        for i, attempt in enumerate(attempts, start=1):
            f.write(
                f"{i:03d}. "
                f"feasible={attempt.get('feasible')}, "
                f"status={attempt.get('status')}, "
                f"epochs={attempt.get('epochs')}, "
                f"batch_size={attempt.get('batch_size')}, "
                f"steps={attempt.get('steps')}, "
                f"q={_fmt_float(attempt.get('q'))}, "
                f"sigma={_fmt_float(attempt.get('sigma'))}, "
                f"noise_scale={_fmt_float(attempt.get('noise_scale'))}, "
                f"eps_high={_fmt_float(attempt.get('eps_high'))}, "
                f"epsilon_gap={_fmt_float(attempt.get('epsilon_gap'))}\n"
            )
    return report_file

def adapt_training_parameters_for_noise(noise_scale, num_epochs, burn_in_epochs, lr0, 
                                       epsilon, delta, n, batch_size, clip_norm,
                                       d=None,
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
        d: Problem dimension (optional, for adaptive bounds)
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
    
    _log(
        f"Starting utility-guided DP calibration search from epochs={num_epochs}, "
        f"batch={batch_size}, initial_noise_scale={noise_scale:.4f}"
    )

    # Compute adaptive epoch bounds based on problem scale (dimension and data size)
    # High-dimensional or small datasets need more careful tuning
    if d is not None:
        min_epochs_adaptive = max(2, int(np.ceil(np.log(d + 1))))  # Log scaling with dimension
        max_epochs_adaptive = max(100, int(np.sqrt(n)))  # Empirical: ~sqrt(n) is often effective
    else:
        min_epochs_adaptive = 2
        max_epochs_adaptive = max(100, int(np.sqrt(n))) if n >= 100 else 50
    
    # Expand search range *adaptively* when noise is very large (e.g., at intermediate epsilon values).
    # Standard range [0.5, 0.75, 1.0, 1.25, 1.5] works well for loose privacy.
    # For high noise (noise_scale > 5), explore more aggressive reductions but limit to ~6 candidates
    # to avoid excessive RDP computations.
    if noise_scale > 8.0:
        # Very high noise: prioritize aggressive epoch reduction
        epoch_factors = [0.1, 0.25, 0.5, 1.0, 1.5]
        min_epochs = min_epochs_adaptive
        _log(f"  Very large noise detected (scale={noise_scale:.2f}); using aggressive epoch reduction factors={epoch_factors}, min_epochs={min_epochs}")
    elif noise_scale > 5.0:
        # Moderately high noise: balance exploration with computation
        epoch_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        min_epochs = min_epochs_adaptive
        _log(f"  Large noise detected (scale={noise_scale:.2f}); expanding epoch search to factors={epoch_factors}, min_epochs={min_epochs}")
    else:
        # Low noise: standard exploration
        epoch_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
        min_epochs = max(5, min_epochs_adaptive)

    epoch_candidates = sorted({
        max(min_epochs, int(round(num_epochs * f))) for f in epoch_factors
    })
    if num_epochs not in epoch_candidates:
        epoch_candidates.append(num_epochs)
        epoch_candidates = sorted(set(epoch_candidates))

    batch_candidates = [batch_size]

    best = None
    attempts = 0

    for candidate_batch in batch_candidates:
        for candidate_epochs in epoch_candidates:
            candidate_steps = candidate_epochs * _steps_per_epoch(n, candidate_batch)
            if candidate_steps <= 0:
                continue

            sigma = compute_noise_multiplier(
                epsilon,
                delta,
                candidate_steps,
                population_size=n,
                sample_size=candidate_batch,
                raise_on_failure=False,
            )
            attempts += 1
            if sigma is None:
                continue

            candidate_noise_scale = sigma * clip_norm / candidate_batch
            # Utility proxy: more updates and lower injected noise are preferred.
            utility_score = candidate_steps / max(candidate_noise_scale ** 2, 1e-12)

            candidate = {
                "epochs": candidate_epochs,
                "batch_size": candidate_batch,
                "steps": candidate_steps,
                "sigma": sigma,
                "noise_scale": candidate_noise_scale,
                "score": utility_score,
            }

            if best is None:
                best = candidate
            else:
                best_key = (best["score"], best["epochs"], best["batch_size"])
                cand_key = (candidate["score"], candidate["epochs"], candidate["batch_size"])
                if cand_key > best_key:
                    best = candidate

    if best is None:
        _log(
            "  ⚠ No feasible candidate found in utility-guided search; "
            "falling back to incoming parameters."
        )
        fallback_steps = num_epochs * _steps_per_epoch(n, batch_size)
        return num_epochs, burn_in_epochs, lr0, fallback_steps, noise_scale

    adapted_epochs = int(best["epochs"])
    adapted_steps = int(best["steps"])
    adapted_noise_scale = float(best["noise_scale"])

    if adapted_epochs != num_epochs:
        adapted_burn_in = max(1, int(round(burn_in_epochs * (adapted_epochs / max(1, num_epochs)))))
    else:
        adapted_burn_in = burn_in_epochs

    _log("Final adaptive parameters (utility-guided):")
    _log(f"  - Epochs: {num_epochs} → {adapted_epochs}")
    _log(f"  - Burn-in: {burn_in_epochs} → {adapted_burn_in}")
    _log(f"  - Learning rate: {lr0:.3f} → {lr0:.3f}")
    _log(f"  - Steps: {num_epochs * _steps_per_epoch(n, batch_size)} → {adapted_steps}")
    _log(f"  - Final noise scale: {noise_scale:.4f} → {adapted_noise_scale:.4f}")

    # Edge case monitoring
    if adapted_epochs < 3:
        _log(f"  ⚠️  WARNING: Very low epochs ({adapted_epochs}); verify gradient convergence and final model quality.")
    if noise_scale > 15.0:
        _log(f"  ⚠️  WARNING: Very high noise (scale={noise_scale:.2f}); consider epsilon ≥ 1.0 for more stable training.")
    
    # Check data utilization
    data_utilization_ratio = (batch_size * adapted_steps) / n
    if data_utilization_ratio < 1.5:
        _log(f"  ⚠️  WARNING: Data underutilized (ratio={data_utilization_ratio:.2f}); only {data_utilization_ratio:.1f}x passes through each example.")
    if data_utilization_ratio > 100:
        _log(f"  ℹ️ High data reuse (ratio={data_utilization_ratio:.2f}); model sees each example ~{data_utilization_ratio:.0f}x.")
    
    if d is not None and adapted_epochs * _steps_per_epoch(n, batch_size) < 10 * d:
        _log(f"  ⚠️  WARNING: Total steps ({adapted_steps}) < 10×d ({10*d}); may be insufficient for d={d}-dimensional estimation.")

    return adapted_epochs, adapted_burn_in, lr0, adapted_steps, adapted_noise_scale

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


def _steps_per_epoch(n, batch_size):
    # Fixed-size shuffled mini-batches with drop-last.
    # Within each epoch, examples are non-overlapping across steps.
    return max(1, n // batch_size)

def compute_noise_multiplier(
    epsilon,
    delta,
    steps,
    population_size=None,
    sample_size=None,
    sigma_bounds=(1e-5, 4e7),
    tol=1e-6,
    raise_on_failure=True,
    return_info=False,
):
    """
    Calibrate Gaussian noise multiplier for fixed-size shuffled mini-batches
    (sampled without replacement, drop-last) using dp-accounting RDP accountant.
    """
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")

    if dp_event is None or rdp is None or privacy_accountant is None:
        msg = "dp-accounting is required for without_replacement accounting; install package 'dp-accounting'."
        if raise_on_failure:
            raise ValueError(msg)
        if return_info:
            return None, {
                "status": "missing_dependency",
                "epsilon_target": epsilon,
                "delta_target": delta,
            }
        return None

    if population_size is None or sample_size is None:
        msg = "population_size and sample_size must be provided for without_replacement accounting."
        if raise_on_failure:
            raise ValueError(msg)
        if return_info:
            return None, {
                "status": "missing_parameters",
                "epsilon_target": epsilon,
                "delta_target": delta,
            }
        return None

    if population_size <= 0:
        raise ValueError(f"population_size must be > 0, got {population_size}")
    if sample_size <= 0 or sample_size > population_size:
        raise ValueError(
            f"sample_size must be in [1, population_size], got sample_size={sample_size}, population_size={population_size}"
        )

    def get_rdp_orders(target_epsilon):
        frac = np.array([1.25, 1.5, 1.75])
        base = np.arange(2.0, 64.0, 1.0)
        if target_epsilon >= 0.3:
            tail = np.array([64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0])
        else:
            tail = np.array([64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0])
        return np.unique(np.concatenate([frac, base, tail]))

    orders = get_rdp_orders(epsilon)

    def compute_epsilon(sigma):
        accountant = rdp.RdpAccountant(
            orders=orders,
            neighboring_relation=privacy_accountant.NeighboringRelation.REPLACE_ONE,
        )
        event = dp_event.SampledWithoutReplacementDpEvent(
            source_dataset_size=int(population_size),
            sample_size=int(sample_size),
            event=dp_event.GaussianDpEvent(float(sigma)),
        )
        accountant.compose(event, count=int(steps))
        return accountant.get_epsilon(delta)

    # Binary search to find sigma such that ε <= target
    ##sigma_min, sigma_max = 1e-3, 4e8

    def objective(sigma):
        return compute_epsilon(sigma) - epsilon

    # Check that the bounds are valid
    eps_low = compute_epsilon(sigma_bounds[0])
    eps_high = compute_epsilon(sigma_bounds[1])



    if eps_low < epsilon and eps_high < epsilon:
        # Entire interval is already private enough; pick smallest sigma for best utility.
        sigma_opt = sigma_bounds[0]
        if return_info:
            return sigma_opt, {
                "status": "over_private_interval",
                "eps_low": eps_low,
                "eps_high": eps_high,
                "epsilon_target": epsilon,
                "delta_target": delta,
            }
        return sigma_opt
        
    if eps_low > epsilon and eps_high > epsilon:
        msg = (
            "n", population_size, "b", sample_size, "steps", steps,
            "sigma_bounds", sigma_bounds, "tol", tol,
            "ε target is too tight: even the largest σ doesn't meet ε.",
            eps_low, eps_high, epsilon, delta
        )
        if raise_on_failure:
            raise ValueError(msg)
        if return_info:
            return None, {
                "status": "infeasible_interval",
                "eps_low": eps_low,
                "eps_high": eps_high,
                "epsilon_target": epsilon,
                "delta_target": delta,
            }
        return None

    # Root-finding for objective(sigma) = 0
    try:
        sigma_opt = brentq(objective, *sigma_bounds, xtol=tol)
    except ValueError:
        if raise_on_failure:
            raise
        if return_info:
            return None, {
                "status": "root_find_failed",
                "eps_low": eps_low,
                "eps_high": eps_high,
                "epsilon_target": epsilon,
                "delta_target": delta,
            }
        return None

    if return_info:
        return sigma_opt, {
            "status": "ok",
            "eps_low": eps_low,
            "eps_high": eps_high,
            "epsilon_target": epsilon,
            "delta_target": delta,
        }
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
#     steps = num_epochs * (n // batch_size)
#     noise_multiplier = compute_noise_multiplier(
#         epsilon, delta, steps, population_size=n, sample_size=batch_size
#     )

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
            g   = grad_log_loss_glm_sgd(theta, X[idx], Y[idx], model) 

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
        run_config   = None,
        save_privacy_search_log = 'on_failure'
):
    """
    DP‑SGD that returns the **average** of the last `tail_batches` parameter
    vectors (post‑processing → same (ε,δ) guarantee).
    
    Args:
        verbose: If True, print DP-SGD messages to stdout
        logger: Optional logging.Logger instance to write messages to log file
        use_default_logger: If True and logger is None, use default logger to write to logs/{run_config}/dp_sgd.log
        run_config: String identifier for the run configuration (creates subfolder logs/{run_config}/)
        save_privacy_search_log: 'on_failure' (default), 'always', or 'never'
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
    local_rng = rng if rng is not None else np.random.default_rng()
    batch_size = min(batch_size, n)  # ensure batch size doesn't exceed dataset

    # — noise multiplier σ chosen via RDP —
    steps = num_epochs * _steps_per_epoch(n, batch_size)
    q = batch_size / n

    initial_num_epochs = num_epochs
    initial_batch_size = batch_size
    attempt_records = []

    def _record_attempt(record):
        attempt_records.append(record)

    def _pick_best_attempt(records):
        if not records:
            return None

        feasible = [r for r in records if r.get("feasible", False)]
        if feasible:
            # Higher epochs and larger batch usually improve utility; then lower noise scale.
            return max(
                feasible,
                key=lambda r: (
                    r.get("epochs", 0),
                    r.get("batch_size", 0),
                    -float('inf') if r.get("noise_scale") is None else -r["noise_scale"],
                ),
            )

        with_gap = [r for r in records if r.get("epsilon_gap") is not None]
        if with_gap:
            return min(
                with_gap,
                key=lambda r: (
                    r["epsilon_gap"],
                    -r.get("epochs", 0),
                    -r.get("batch_size", 0),
                ),
            )
        return records[-1]

    def _try_calibration(candidate_epochs, candidate_batch_size):
        candidate_steps = candidate_epochs * _steps_per_epoch(n, candidate_batch_size)
        candidate_q = candidate_batch_size / n
        sigma_val, sigma_info = compute_noise_multiplier(
            epsilon,
            delta,
            candidate_steps,
            population_size=n,
            sample_size=candidate_batch_size,
            raise_on_failure=False,
            return_info=True,
        )
        status = sigma_info.get("status", "unknown") if sigma_info else "unknown"
        eps_high = sigma_info.get("eps_high") if sigma_info else None
        epsilon_gap = None if eps_high is None else max(0.0, eps_high - epsilon)

        if sigma_val is None:
            _record_attempt({
                "feasible": False,
                "status": status,
                "epochs": candidate_epochs,
                "batch_size": candidate_batch_size,
                "steps": candidate_steps,
                "q": candidate_q,
                "sigma": None,
                "noise_scale": None,
                "eps_high": eps_high,
                "epsilon_gap": epsilon_gap,
            })
            _log(
                f"  Infeasible privacy calibration: epochs={candidate_epochs}, "
                f"batch={candidate_batch_size}, steps={candidate_steps}, q={candidate_q:.6f}, status={status}"
            )
            return None
        candidate_noise_scale = sigma_val * clip_norm / candidate_batch_size
        _record_attempt({
            "feasible": True,
            "status": status,
            "epochs": candidate_epochs,
            "batch_size": candidate_batch_size,
            "steps": candidate_steps,
            "q": candidate_q,
            "sigma": sigma_val,
            "noise_scale": candidate_noise_scale,
            "eps_high": eps_high,
            "epsilon_gap": epsilon_gap,
        })
        _log(
            f"  Feasible calibration: epochs={candidate_epochs}, batch={candidate_batch_size}, "
            f"steps={candidate_steps}, q={candidate_q:.6f}, sigma={sigma_val:.4f}, "
            f"noise_scale={candidate_noise_scale:.4f}"
        )
        return {
            "epochs": candidate_epochs,
            "batch_size": candidate_batch_size,
            "steps": candidate_steps,
            "q": candidate_q,
            "sigma": sigma_val,
            "noise_scale": candidate_noise_scale,
        }

    calibration = _try_calibration(num_epochs, batch_size)

    # If initial config is infeasible, search over privacy-relevant knobs only.
    if calibration is None:
        _log("⚠ Initial privacy calibration infeasible. Searching with binary search over epochs across batch-size candidates...")
        original_epochs = num_epochs
        original_batch_size = batch_size
        feasible_candidates = []

        raw_batch_candidates = []
        for batch_factor in [1.0, 0.75, 0.5, 0.25, 0.1]:
            candidate_batch = max(1, int(original_batch_size * batch_factor))
            candidate_batch = min(candidate_batch, n)
            raw_batch_candidates.append(candidate_batch)

        # Deduplicate collapsed candidates (common when n is tiny).
        batch_candidates = list(dict.fromkeys(raw_batch_candidates))

        if len(batch_candidates) < len(raw_batch_candidates):
            _log(
                f"  Deduplicated batch candidates from {len(raw_batch_candidates)} to "
                f"{len(batch_candidates)}: {batch_candidates}"
            )

        for candidate_batch in batch_candidates:
            _log(f"  Binary-searching epochs for batch={candidate_batch}")

            lo, hi = 1, max(1, original_epochs)
            best_for_batch = None
            seen_mid = set()

            while lo <= hi:
                mid = (lo + hi) // 2
                if mid in seen_mid:
                    break
                seen_mid.add(mid)

                candidate = _try_calibration(mid, candidate_batch)
                if candidate is not None:
                    best_for_batch = candidate
                    lo = mid + 1  # push toward higher utility (more epochs)
                else:
                    hi = mid - 1

            if best_for_batch is not None:
                feasible_candidates.append(best_for_batch)

        if feasible_candidates:
            calibration = max(
                feasible_candidates,
                key=lambda c: (c["epochs"], c["batch_size"], -c["noise_scale"]),
            )

    best_attempt = _pick_best_attempt(attempt_records)
    if best_attempt is not None:
        _log(
            "Best calibration attempt tried: "
            f"feasible={best_attempt.get('feasible')}, "
            f"epochs={best_attempt.get('epochs')}, batch={best_attempt.get('batch_size')}, "
            f"steps={best_attempt.get('steps')}, q={best_attempt.get('q')}, "
            f"sigma={best_attempt.get('sigma')}, noise_scale={best_attempt.get('noise_scale')}, "
            f"status={best_attempt.get('status')}, epsilon_gap={best_attempt.get('epsilon_gap')}"
        )

    report = {
        "epsilon": epsilon,
        "delta": delta,
        "run_config": run_config,
        "initial_epochs": initial_num_epochs,
        "initial_batch_size": initial_batch_size,
        "attempt_count": len(attempt_records),
        "best_attempt": best_attempt,
        "selected_calibration": calibration,
        "attempts": attempt_records,
    }

    report_file = None
    if save_privacy_search_log not in ('on_failure', 'always', 'never'):
        save_privacy_search_log = 'on_failure'

    should_write_report = (
        save_privacy_search_log == 'always' or
        (save_privacy_search_log == 'on_failure' and calibration is None)
    )

    if should_write_report:
        report_file = _write_privacy_search_log(run_config, report)
        _log(f"Saved privacy calibration report to {report_file}")

    if calibration is None:
        _log(
            "⚠ No feasible private training configuration found under fixed ε,δ. "
            "Raising error (no silent fallback to old parameters)."
        )
        report_hint = (
            f" All calibration attempts are stored in: {report_file}"
            if report_file is not None
            else ""
        )
        raise ValueError(
            "No feasible private training configuration found under fixed ε,δ "
            f"after {len(attempt_records)} calibration attempts. "
            f"Best attempt summary: {best_attempt}. "
            f"save_privacy_search_log={save_privacy_search_log}."
            f"{report_hint}"
        )

    sigma = calibration["sigma"]
    noise_scale = calibration["noise_scale"]
    q = calibration["q"]
    steps = calibration["steps"]
    selected_epochs = calibration["epochs"]
    selected_batch_size = calibration["batch_size"]

    if selected_epochs != num_epochs:
        burn_in_epochs = max(1, int(burn_in_epochs * (selected_epochs / max(1, num_epochs))))
    num_epochs = selected_epochs
    batch_size = selected_batch_size

    _log(
        f"DP-SGD selected config: ε={epsilon}, δ={delta}, epochs={num_epochs}, "
        f"batch={batch_size}, steps={steps}, q={q:.6f}, σ={sigma:.4f}, "
        f"noise_scale={noise_scale:.4f}"
    )

    # Extract dimension for adaptive bound computation
    d = X.shape[1] if len(X.shape) > 1 else 1
    
    # Compute advisory batch size for this dataset (not used in current path, but useful for future extensions)
    # Rule: aim for 20-100 batches per epoch for stable gradient averaging
    advise_batches_per_epoch = max(20, min(100, n // 1000))  # Heuristic scaling
    advised_batch_size = max(32, n // advise_batches_per_epoch)
    if advised_batch_size != batch_size and verbose:
        _log(f"ℹ️ Advisory batch size for dataset n={n}: {advised_batch_size} (current: {batch_size})")
    
    # Adaptive parameter adjustment for large noise scales (with dimension-aware bounds)
    num_epochs, burn_in_epochs, lr0, steps, noise_scale = adapt_training_parameters_for_noise(
        noise_scale,
        num_epochs,
        burn_in_epochs,
        lr0,
        epsilon,
        delta,
        n,
        batch_size,
        clip_norm,
        d=d,
        verbose=verbose,
        logger=logger,
        use_default_logger=use_default_logger,
        run_config=run_config,
    )
    # Recalibrate sigma after adaptation so the training loop uses the
    # accountant-consistent noise for the final (epochs, batch_size, steps).
    q = batch_size / n
    steps = num_epochs * _steps_per_epoch(n, batch_size)
    sigma_after_adapt = compute_noise_multiplier(
        epsilon,
        delta,
        steps,
        population_size=n,
        sample_size=batch_size,
        raise_on_failure=True,
    )
    sigma = sigma_after_adapt
    noise_scale = sigma * clip_norm / batch_size

    if num_epochs <= 0:
        _log("⚠ Adaptation reduced epochs to 0; returning initial parameters.")
        return theta, True
    t_step = 0
    steps_per_epoch = _steps_per_epoch(n, batch_size)
    idx_all = np.arange(n)
    theta_avg = np.zeros_like(theta, dtype=float)
    k=0

    for epoch in range(num_epochs):
        local_rng.shuffle(idx_all)
        for start in range(0, steps_per_epoch * batch_size, batch_size):
            t_step += 1
            lr_t = lr0 / np.sqrt(t_step)

            idx = idx_all[start:start + batch_size]
            X_b, Y_b = X[idx], Y[idx]

            # per‑sample gradients + clipping
            grads = []
            for x_i, y_i in zip(X_b, Y_b):
                g_i = grad_log_loss_glm_sgd(theta, x_i[None, :], np.array([y_i]), model)
                g_i_norm = np.linalg.norm(g_i)
                if g_i_norm > clip_norm:
                    g_i *= clip_norm / g_i_norm
                grads.append(g_i)

            g_bar = np.mean(grads, axis=0) + 2.0 * lmbda * theta
            noise = local_rng.normal(0.0, sigma * clip_norm / batch_size, size=d)
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
        

    # post‑processing: average of post-burn-in epoch snapshots
    if k == 0:
        _log("⚠ No post-burn-in iterates collected; returning final iterate instead of averaged iterate.")
        return theta, True
    return theta_avg/k, True


