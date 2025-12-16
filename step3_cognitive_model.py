"""
Cognitive Model of Password Editing Behavior
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = 'dataset_enriched.csv'
OUTPUT_DIR = Path('model_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("STEP 3: COGNITIVE MODEL OF PASSWORD EDITING BEHAVIOR")
print("="*70)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data():
    """
    Load enriched dataset and prepare for modeling.
    Focus on top 5 strategies that cover 80%+ of data.
    """
    print("\n--- Loading Data ---")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} password edits")

    # Define strategy set based on EDA (top 5 + "other" catchall)
    top_strategies = [
        'targeted_deletion',
        'leetspeak_substitution',
        'complete_rewrite',
        'append_suffix',
        'insert_middle'
    ]

    # Collapse rare strategies into "other"
    df['strategy_simplified'] = df['L3_primary_strategy'].apply(
        lambda s: s if s in top_strategies else 'other'
    )

    # Create strategy index (for discrete choice modeling)
    strategy_to_idx = {s: i for i, s in enumerate(top_strategies + ['other'])}
    df['strategy_idx'] = df['strategy_simplified'].map(strategy_to_idx)

    print(f"\nStrategy distribution (simplified):")
    print(df['strategy_simplified'].value_counts())

    # Extract features for modeling
    features = {
        'strategy_idx': df['strategy_idx'].values,
        'task': df['Task'].values,
        'security': df['security_guesses_log10'].fillna(df['security_guesses_log10'].mean()).values,
        'memory_cost': df['memory_cost_estimate'].values,
        'additive_ops': df['L2_additive_ops_count'].values,
        'subtractive_ops': df['L2_subtractive_ops_count'].values,
        'person_id': df['person_id'].values,
    }

    return df, features, strategy_to_idx, top_strategies + ['other']


# ============================================================================
# COGNITIVE MODEL DEFINITION
# ============================================================================

class PasswordEditingModel:
    """
    Utility-based model of password editing strategy selection.

    Parameters:
        alpha_security: Weight on security improvement
        alpha_memory: Weight on memory cost (negative utility)
        beta_strategies: Base preference for each strategy (6-dim vector)
        lambda_constraint: Effect of deletion-only constraint (Task 2)
        lambda_time: Effect of time pressure (Task 4)
        tau: Rationality/inverse temperature parameter
    """

    def __init__(self, num_strategies=6):
        self.num_strategies = num_strategies
        self.strategy_names = None  # Will be set later

    def utility(self,
                security: float,
                memory_cost: float,
                strategy_idx: int,
                task: int,
                alpha_security: float,
                alpha_memory: float,
                beta_strategies: np.ndarray,
                lambda_constraint: float,
                lambda_time: float) -> float:
        """
        Compute utility of a password edit with given strategy.

        U(edit) = α_security × Security - α_memory × MemoryCost + β_strategy + ε

        Constraints:
        - Task 2 (deletion-only): Suppress additive strategies
        - Task 4 (time pressure): Add time cost
        """
        # Base utility: security-memory tradeoff
        base_utility = (alpha_security * security -
                       alpha_memory * memory_cost)

        # Strategy preference
        strategy_utility = beta_strategies[strategy_idx]

        # Constraint effects
        constraint_penalty = 0.0

        # Task 2: Deletion-only constraint (penalize additive strategies)
        if task == 2:
            # Penalize non-subtractive strategies
            # Strategy indices: 0=deletion (OK), 1=leetspeak (OK-ish), 2=rewrite (OK),
            #                   3=append (BAD), 4=insert (BAD), 5=other
            if strategy_idx in [3, 4]:  # append, insert are additive
                constraint_penalty -= lambda_constraint  # Large penalty

        # Task 4: Time pressure (favor simpler strategies)
        if task == 4:
            # Penalize complex strategies (rewrite is expensive)
            if strategy_idx == 2:  # complete_rewrite
                constraint_penalty -= lambda_time

        total_utility = base_utility + strategy_utility + constraint_penalty

        return total_utility

    def choice_probability(self,
                          strategy_idx: int,
                          security: float,
                          memory_cost: float,
                          task: int,
                          params: Dict) -> float:
        """
        Compute probability of choosing a strategy via softmax.

        P(strategy | context) = exp(τ × U(strategy)) / Σ exp(τ × U(s'))
        """
        tau = params['tau']

        # Compute utility for chosen strategy
        u_chosen = self.utility(
            security, memory_cost, strategy_idx, task,
            params['alpha_security'], params['alpha_memory'],
            params['beta_strategies'], params['lambda_constraint'],
            params['lambda_time']
        )

        # Compute utilities for all strategies (for normalization)
        utilities = np.array([
            self.utility(
                security, memory_cost, s, task,
                params['alpha_security'], params['alpha_memory'],
                params['beta_strategies'], params['lambda_constraint'],
                params['lambda_time']
            )
            for s in range(self.num_strategies)
        ])

        # Softmax with rationality parameter
        log_probs = tau * utilities
        log_probs = log_probs - np.max(log_probs)  # Numerical stability
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs)

        return probs[strategy_idx]

    def log_likelihood(self, data: Dict, params: Dict) -> float:
        """
        Compute log-likelihood of observed data under model.

        L(params | data) = Σ log P(strategy_i | context_i, params)
        """
        n = len(data['strategy_idx'])
        log_lik = 0.0

        for i in range(n):
            prob = self.choice_probability(
                data['strategy_idx'][i],
                data['security'][i],
                data['memory_cost'][i],
                data['task'][i],
                params
            )
            # Add small epsilon to avoid log(0)
            log_lik += np.log(prob + 1e-10)

        return log_lik

    def negative_log_likelihood(self, param_vec: np.ndarray, data: Dict) -> float:
        """
        Negative log-likelihood for optimization (to minimize).

        param_vec: [alpha_security, alpha_memory, beta_0, ..., beta_5,
                    lambda_constraint, lambda_time, tau]
        """
        # Unpack parameters
        params = self.vec_to_params(param_vec)

        # Compute log-likelihood
        ll = self.log_likelihood(data, params)

        # Return negative (for minimization)
        return -ll

    def vec_to_params(self, param_vec: np.ndarray) -> Dict:
        """
        Convert parameter vector to dictionary.
        Uses normalization constraint: alpha_memory = 1 - alpha_security

        param_vec: [alpha_security, beta_0, ..., beta_5, tau]
        """
        alpha_security = param_vec[0]
        alpha_memory = 1.0 - alpha_security  # Normalization constraint

        return {
            'alpha_security': alpha_security,
            'alpha_memory': alpha_memory,
            'beta_strategies': param_vec[1:7],  # 6 strategies
            'lambda_constraint': 0.0,  # Not used
            'lambda_time': 0.0,        # Not used
            'tau': param_vec[7]
        }

    def params_to_vec(self, params: Dict) -> np.ndarray:
        """
        Convert parameter dictionary to vector.
        Only includes alpha_security (alpha_memory is derived)
        """
        return np.concatenate([
            [params['alpha_security']],
            params['beta_strategies'],
            [params['tau']]
        ])


# ============================================================================
# MODEL FITTING
# ============================================================================

def initialize_parameters(strategy_counts: Dict) -> Dict:
    """
    Initialize parameters based on EDA findings and prior knowledge.
    """
    print("\n--- Initializing Parameters ---")

    # Strategy preferences based on observed frequencies
    # More common strategies get higher initial beta values
    total = sum(strategy_counts.values())
    beta_strategies = np.array([
        np.log(strategy_counts.get('targeted_deletion', 1) / total),      # 0
        np.log(strategy_counts.get('leetspeak_substitution', 1) / total),  # 1
        np.log(strategy_counts.get('complete_rewrite', 1) / total),        # 2
        np.log(strategy_counts.get('append_suffix', 1) / total),           # 3
        np.log(strategy_counts.get('insert_middle', 1) / total),           # 4
        np.log(strategy_counts.get('other', 1) / total),                   # 5
    ])

    # Initialize with normalized weights (sum to 1)
    # Start with equal weights for security and memory
    params = {
        'alpha_security': 0.5,      # 50% weight on security (normalized from 1.0)
        'alpha_memory': 0.5,        # 50% weight on memory (derived: 1 - 0.5)
        'beta_strategies': beta_strategies,
        'lambda_constraint': 0.0,   # Not used (hard constraint for Task 2 instead)
        'lambda_time': 0.0,         # Not used (will see effect in β and τ)
        'tau': 2.0,                 # Moderate rationality
    }

    print("Initial parameters (normalized: α_security + α_memory = 1):")
    print("  Starting with equal weights (both = 1.0 before normalization)")
    for key, val in params.items():
        if key == 'beta_strategies':
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: {val:.3f}")
    print(f"  Ratio (security/memory): {params['alpha_security']/params['alpha_memory']:.2f}")

    return params


def fit_model_mle(model: PasswordEditingModel,
                  data: Dict,
                  initial_params: Dict,
                  max_iter: int = 500) -> Tuple[Dict, float]:
    """
    Fit model parameters using Maximum Likelihood Estimation.
    Uses scipy.optimize for gradient-based optimization.
    """
    print("\n--- Fitting Model (MLE with scipy.optimize) ---")

    # Convert initial params to vector
    param_vec = model.params_to_vec(initial_params)

    # Define loss function (negative log-likelihood)
    def loss_fn(params):
        # Ensure tau > 0 and alpha_security in valid range
        if params[7] <= 0:  # tau is now at index 7
            return 1e10  # Large penalty for invalid tau
        if params[0] < 0.01 or params[0] > 0.99:  # alpha_security
            return 1e10  # Large penalty for invalid alpha
        return model.negative_log_likelihood(params, data)

    # Callback to print progress
    iteration = [0]
    def callback(params):
        if iteration[0] % 50 == 0:
            loss = loss_fn(params)
            print(f"Iteration {iteration[0]:3d} | Neg-Log-Lik: {loss:10.2f}")
        iteration[0] += 1

    print("Starting optimization...")
    print("-" * 50)

    # Optimize using L-BFGS-B (allows bounds)
    # Reduced parameter vector: [alpha_security, beta_0, ..., beta_5, tau]
    # alpha_memory = 1 - alpha_security (enforced in vec_to_params)
    bounds = [
        (0.01, 0.99),  # alpha_security (0.01 to 0.99, since alpha_memory = 1 - alpha_security)
        (-5, 5),       # beta_0 (targeted_deletion)
        (-5, 5),       # beta_1 (leetspeak_substitution)
        (-5, 5),       # beta_2 (complete_rewrite)
        (-5, 5),       # beta_3 (append_suffix)
        (-5, 5),       # beta_4 (insert_middle)
        (-5, 5),       # beta_5 (other)
        (0.1, 10),     # tau (must be positive)
    ]

    result = minimize(
        loss_fn,
        param_vec,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'disp': False},
        callback=callback
    )

    final_params = model.vec_to_params(result.x)
    final_loss = result.fun

    print("-" * 50)
    print(f"Optimization {'converged' if result.success else 'did not converge'}")
    print(f"Final negative log-likelihood: {final_loss:.2f}")

    return final_params, final_loss


def fit_model_mle_with_constraint(model: PasswordEditingModel,
                                   data: Dict,
                                   initial_params: Dict,
                                   max_iter: int = 500) -> Tuple[Dict, float]:
    """
    Fit model for Task 2 with hard deletion-only constraint.
    Sets β = -∞ for leetspeak, complete_rewrite, append, and insert.
    Only deletion and other (minimal changes) are allowed.
    Only fits α_security, α_memory, β_deletion, β_other, and τ.
    """
    print("\n--- Fitting Model with Hard Constraint (MLE) ---")

    # For Task 2: only deletion and minimal changes are possible
    # Strategy indices: 0=deletion, 1=leetspeak, 2=rewrite, 3=append, 4=insert, 5=other
    # Set leetspeak (1), rewrite (2), append (3), and insert (4) to -∞ (impossible)

    # Reduced parameter vector: [alpha_security, beta_0, beta_5, tau]
    # (exclude beta_1, beta_2, beta_3, beta_4 - all disallowed strategies)
    # alpha_memory = 1 - alpha_security (normalization)

    def loss_fn(params_reduced):
        # Reconstruct full parameter vector with constraints
        alpha_security = params_reduced[0]
        alpha_memory = 1.0 - alpha_security  # Normalization

        full_params = np.zeros(8)
        full_params[0] = alpha_security
        full_params[1] = params_reduced[1]  # beta_0 (deletion)
        full_params[2] = -100.0  # beta_1 (leetspeak) - effectively -∞
        full_params[3] = -100.0  # beta_2 (rewrite) - effectively -∞
        full_params[4] = -100.0  # beta_3 (append) - effectively -∞
        full_params[5] = -100.0  # beta_4 (insert) - effectively -∞
        full_params[6] = params_reduced[2]  # beta_5 (other)
        full_params[7] = params_reduced[3]  # tau

        if full_params[7] <= 0 or alpha_security < 0.01 or alpha_security > 0.99:
            return 1e10
        return model.negative_log_likelihood(full_params, data)

    # Initial parameters (reduced)
    # For Task 2, start with better initial guesses
    initial_vec = np.array([
        initial_params['alpha_security'],  # Will be normalized
        2.0,   # deletion - start higher since it's the main strategy
        0.0,   # other - minimal_change maps here
        4.0    # tau - expect high rationality for constrained task
    ])

    # Bounds for reduced parameters
    bounds = [
        (0.01, 0.99),  # alpha_security (alpha_memory = 1 - alpha_security)
        (-5, 5),       # beta_0 (deletion)
        (-5, 5),       # beta_5 (other)
        (0.1, 10),     # tau
    ]

    # Callback
    iteration = [0]
    def callback(params):
        if iteration[0] % 50 == 0:
            loss = loss_fn(params)
            print(f"Iteration {iteration[0]:3d} | Neg-Log-Lik: {loss:10.2f}")
        iteration[0] += 1

    print("Starting optimization...")
    print("-" * 50)

    result = minimize(
        loss_fn,
        initial_vec,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'disp': False},
        callback=callback
    )

    # Reconstruct full parameter dictionary
    params_reduced = result.x
    alpha_security = params_reduced[0]
    alpha_memory = 1.0 - alpha_security  # Normalization

    final_params = {
        'alpha_security': alpha_security,
        'alpha_memory': alpha_memory,
        'beta_strategies': np.array([
            params_reduced[1],  # deletion
            -100.0,             # leetspeak (constrained)
            -100.0,             # rewrite (constrained)
            -100.0,             # append (constrained)
            -100.0,             # insert (constrained)
            params_reduced[2]   # other
        ]),
        'lambda_constraint': 0.0,  # Not applicable
        'lambda_time': 0.0,        # Not applicable
        'tau': params_reduced[3]
    }

    final_loss = result.fun

    print("-" * 50)
    print(f"Optimization {'converged' if result.success else 'did not converge'}")
    print(f"Final negative log-likelihood: {final_loss:.2f}")

    return final_params, final_loss


def cross_validate_model(model: PasswordEditingModel,
                         data: Dict,
                         initial_params: Dict,
                         task_num: int,
                         k_folds: int = 5) -> Dict:
    """
    Perform k-fold cross-validation for a single task.

    Returns:
        Dictionary with in-sample and out-of-sample metrics
    """
    print(f"\n  Running {k_folds}-fold cross-validation...")

    # Get indices for this task
    task_mask = data['task'] == task_num
    task_indices = np.where(task_mask)[0]
    n = len(task_indices)

    # Shuffle indices
    np.random.seed(42)  # For reproducibility
    shuffled_indices = np.random.permutation(task_indices)

    # Create folds
    fold_size = n // k_folds

    in_sample_losses = []
    out_sample_losses = []
    out_sample_accuracies = []

    for fold in range(k_folds):
        # Define train/test split
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < k_folds - 1 else n

        test_indices = shuffled_indices[test_start:test_end]
        train_indices = np.concatenate([
            shuffled_indices[:test_start],
            shuffled_indices[test_end:]
        ])

        # Create train and test data
        train_data = {k: v[train_indices] for k, v in data.items()}
        test_data = {k: v[test_indices] for k, v in data.items()}

        # Fit model on training data
        if task_num == 2:
            params, train_loss = fit_model_mle_with_constraint(
                model, train_data, initial_params, max_iter=300
            )
        else:
            params, train_loss = fit_model_mle(
                model, train_data, initial_params, max_iter=300
            )

        # Evaluate on test data
        test_loss = model.negative_log_likelihood(
            model.params_to_vec(params), test_data
        )

        # Compute prediction accuracy on test data
        correct = 0
        for i in range(len(test_data['strategy_idx'])):
            # Get predicted probabilities
            probs = np.array([
                model.choice_probability(
                    s,
                    test_data['security'][i],
                    test_data['memory_cost'][i],
                    test_data['task'][i],
                    params
                )
                for s in range(model.num_strategies)
            ])

            # Predicted strategy is argmax
            predicted = np.argmax(probs)
            actual = test_data['strategy_idx'][i]

            if predicted == actual:
                correct += 1

        accuracy = correct / len(test_data['strategy_idx'])

        in_sample_losses.append(train_loss)
        out_sample_losses.append(test_loss)
        out_sample_accuracies.append(accuracy)

    print(f"  In-sample neg-LL: {np.mean(in_sample_losses):.2f} ± {np.std(in_sample_losses):.2f}")
    print(f"  Out-of-sample neg-LL: {np.mean(out_sample_losses):.2f} ± {np.std(out_sample_losses):.2f}")
    print(f"  Out-of-sample accuracy: {np.mean(out_sample_accuracies):.3f} ± {np.std(out_sample_accuracies):.3f}")

    return {
        'in_sample_loss_mean': np.mean(in_sample_losses),
        'in_sample_loss_std': np.std(in_sample_losses),
        'out_sample_loss_mean': np.mean(out_sample_losses),
        'out_sample_loss_std': np.std(out_sample_losses),
        'out_sample_acc_mean': np.mean(out_sample_accuracies),
        'out_sample_acc_std': np.std(out_sample_accuracies),
    }


def fit_model_by_task(model: PasswordEditingModel,
                      data: Dict,
                      initial_params: Dict,
                      do_cv: bool = True) -> Dict:
    """
    Fit separate models for each individual task.
    Compare Task 1 vs Task 2 (deletion constraint effect)
    and Task 3 vs Task 4 (time pressure effect).

    Task 2 uses hard constraints (β=-∞ for additive strategies).

    Args:
        do_cv: If True, also perform cross-validation
    """
    print("\n" + "="*70)
    print("FITTING SEPARATE MODELS FOR EACH TASK")
    print("="*70)

    results = {}

    for task_num in [1, 2, 3, 4]:
        print(f"\n### Task {task_num} ###")

        # Filter data for this task
        task_mask = data['task'] == task_num
        task_data = {k: v[task_mask] for k, v in data.items()}

        # For Task 2, use hard constraint (deletion-only)
        if task_num == 2:
            print("  Applying hard deletion-only constraint (β=-∞ for additive strategies)")
            params, loss = fit_model_mle_with_constraint(model, task_data, initial_params)
        else:
            # Fit model normally
            params, loss = fit_model_mle(model, task_data, initial_params)

        results[f'task_{task_num}'] = {
            'params': params,
            'loss': loss,
            'n': np.sum(task_mask)
        }

        # Cross-validation
        if do_cv:
            cv_results = cross_validate_model(
                model, data, initial_params, task_num, k_folds=5
            )
            results[f'task_{task_num}']['cv'] = cv_results

    return results


# ============================================================================
# MODEL ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_fitted_parameters(results: Dict, strategy_names: List[str]):
    """
    Analyze and interpret fitted parameters.
    Compare Task 1 vs 2 (deletion constraint) and Task 3 vs 4 (time pressure).
    """
    print("\n" + "="*70)
    print("FITTED PARAMETER ANALYSIS")
    print("="*70)

    # Display individual task results
    for task_num in [1, 2, 3, 4]:
        task_key = f'task_{task_num}'
        res = results[task_key]
        params = res['params']

        print(f"\n### TASK {task_num} ###")
        print(f"Data points: {res['n']}")
        print(f"Neg-log-likelihood: {res['loss']:.2f}")

        # Display cross-validation results if available
        if 'cv' in res:
            cv = res['cv']
            print(f"\n5-Fold Cross-Validation:")
            print(f"  In-sample neg-LL:       {cv['in_sample_loss_mean']:.2f} ± {cv['in_sample_loss_std']:.2f}")
            print(f"  Out-of-sample neg-LL:   {cv['out_sample_loss_mean']:.2f} ± {cv['out_sample_loss_std']:.2f}")
            print(f"  Out-of-sample accuracy: {cv['out_sample_acc_mean']:.3f} ± {cv['out_sample_acc_std']:.3f}")
        print()

        print("Parameters:")
        print(f"  α_security (security weight):      {params['alpha_security']:.3f}")
        print(f"  α_memory (memory cost weight):     {params['alpha_memory']:.3f}")
        print(f"  Security/Memory ratio:             {params['alpha_security']/params['alpha_memory']:.2f}")
        print(f"\n  λ_constraint (deletion penalty):   {params['lambda_constraint']:.3f}")
        print(f"  λ_time (time pressure penalty):    {params['lambda_time']:.3f}")
        print(f"  τ (rationality):                   {params['tau']:.3f}")

        print(f"\n  Strategy preferences (β):")
        for i, name in enumerate(strategy_names):
            print(f"    {name:25s}: {params['beta_strategies'][i]:7.3f}")

    # Compare Task 1 vs Task 2 (Deletion Constraint Effect)
    print("\n" + "="*70)
    print("COMPARISON: TASK 1 vs TASK 2 (Deletion Constraint Effect)")
    print("="*70)

    params_1 = results['task_1']['params']
    params_2 = results['task_2']['params']

    print("\nParameter Changes (Task 2 - Task 1):")
    print(f"  Δα_security:      {params_2['alpha_security'] - params_1['alpha_security']:+7.3f}")
    print(f"  Δα_memory:        {params_2['alpha_memory'] - params_1['alpha_memory']:+7.3f}")
    print(f"  Δλ_constraint:    {params_2['lambda_constraint'] - params_1['lambda_constraint']:+7.3f}")
    print(f"  Δτ (rationality): {params_2['tau'] - params_1['tau']:+7.3f}")

    print(f"\n  Strategy preference changes (Δβ):")
    for i, name in enumerate(strategy_names):
        delta = params_2['beta_strategies'][i] - params_1['beta_strategies'][i]
        print(f"    {name:25s}: {delta:+7.3f}")

    # Compare Task 3 vs Task 4 (Time Pressure Effect)
    print("\n" + "="*70)
    print("COMPARISON: TASK 3 vs TASK 4 (Time Pressure Effect)")
    print("="*70)

    params_3 = results['task_3']['params']
    params_4 = results['task_4']['params']

    print("\nParameter Changes (Task 4 - Task 3):")
    print(f"  Δα_security:      {params_4['alpha_security'] - params_3['alpha_security']:+7.3f}")
    print(f"  Δα_memory:        {params_4['alpha_memory'] - params_3['alpha_memory']:+7.3f}")
    print(f"  Δλ_time:          {params_4['lambda_time'] - params_3['lambda_time']:+7.3f}")
    print(f"  Δτ (rationality): {params_4['tau'] - params_3['tau']:+7.3f}")

    print(f"\n  Strategy preference changes (Δβ):")
    for i, name in enumerate(strategy_names):
        delta = params_4['beta_strategies'][i] - params_3['beta_strategies'][i]
        print(f"    {name:25s}: {delta:+7.3f}")


def visualize_model_predictions(model: PasswordEditingModel,
                                results: Dict,
                                data: Dict,
                                strategy_names: List[str]):
    """
    Visualize model predictions vs. observed data for each task.
    """
    print("\n--- Generating Visualizations ---")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Predictions vs. Observed Data (By Task)', fontsize=16, fontweight='bold')

    for task_num in [1, 2, 3, 4]:
        row = (task_num - 1) // 2
        col = (task_num - 1) % 2
        ax = axes[row, col]

        task_key = f'task_{task_num}'
        params = results[task_key]['params']

        # Filter data for this task
        task_mask = data['task'] == task_num
        task_data = {k: v[task_mask] for k, v in data.items()}

        # Observed frequencies
        observed = np.bincount(task_data['strategy_idx'],
                              minlength=len(strategy_names))
        observed_pct = 100 * observed / np.sum(observed)

        # Predicted probabilities (average over data points)
        predicted = np.zeros(len(strategy_names))
        for i in range(len(task_data['strategy_idx'])):
            for s in range(len(strategy_names)):
                prob = model.choice_probability(
                    s,
                    task_data['security'][i],
                    task_data['memory_cost'][i],
                    task_num,
                    params
                )
                predicted[s] += prob
        predicted_pct = 100 * predicted / len(task_data['strategy_idx'])

        # Plot
        x = np.arange(len(strategy_names))
        width = 0.35

        ax.bar(x - width/2, observed_pct, width, label='Observed', alpha=0.7, color='steelblue')
        ax.bar(x + width/2, predicted_pct, width, label='Predicted', alpha=0.7, color='coral')

        ax.set_ylabel('Percentage', fontsize=11)

        # Task labels with descriptions
        task_labels = {
            1: 'Task 1 (No Restrictions)',
            2: 'Task 2 (Deletion Only)',
            3: 'Task 3 (No Restrictions)',
            4: 'Task 4 (Time Pressure)'
        }
        ax.set_title(task_labels[task_num], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in strategy_names],
                          rotation=0, fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(100, max(observed_pct.max(), predicted_pct.max()) * 1.1))

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'model_predictions_vs_observed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def visualize_parameter_comparison(results: Dict, strategy_names: List[str]):
    """
    Create bar charts comparing fitted parameters across tasks.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    task_nums = [1, 2, 3, 4]
    task_labels = ['Task 1\n(No Restrictions)', 'Task 2\n(Deletion Only)',
                   'Task 3\n(No Restrictions)', 'Task 4\n(Time Pressure)']

    # Extract parameters
    alpha_security = [results[f'task_{i}']['params']['alpha_security'] for i in task_nums]
    alpha_memory = [results[f'task_{i}']['params']['alpha_memory'] for i in task_nums]
    tau = [results[f'task_{i}']['params']['tau'] for i in task_nums]

    # 1. Alpha parameters (stacked bar)
    ax = axes[0, 0]
    x = np.arange(len(task_nums))
    width = 0.5

    ax.bar(x, alpha_security, width, label='α_security', color='darkred', alpha=0.7)
    ax.bar(x, alpha_memory, width, bottom=alpha_security, label='α_memory', color='darkblue', alpha=0.7)

    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Utility Weights (Normalized)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # 2. Tau (rationality)
    ax = axes[0, 1]
    bars = ax.bar(x, tau, width, color=['steelblue', 'orange', 'steelblue', 'green'], alpha=0.7)

    # Highlight the dramatic increase in Task 2
    bars[1].set_edgecolor('black')
    bars[1].set_linewidth(2)

    ax.set_ylabel('τ (Rationality)', fontsize=12)
    ax.set_title('Rationality Parameter Across Tasks', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, tau)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Strategy preferences (β) - Heatmap
    ax = axes[1, 0]
    beta_matrix = np.array([results[f'task_{i}']['params']['beta_strategies'] for i in task_nums])

    im = ax.imshow(beta_matrix.T, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_yticks(np.arange(len(strategy_names)))
    ax.set_yticklabels([s.replace('_', ' ').title() for s in strategy_names], fontsize=9)
    ax.set_title('Strategy Preferences (β) Heatmap', fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('β value', fontsize=11)

    # Add text annotations
    for i in range(len(task_nums)):
        for j in range(len(strategy_names)):
            text = ax.text(i, j, f'{beta_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    # 4. Cross-validation accuracy
    ax = axes[1, 1]

    cv_means = []
    cv_stds = []
    for i in task_nums:
        res = results[f'task_{i}']
        if 'cv' in res:
            cv_means.append(res['cv']['out_sample_acc_mean'])
            cv_stds.append(res['cv']['out_sample_acc_std'])
        else:
            cv_means.append(0)
            cv_stds.append(0)

    bars = ax.bar(x, cv_means, width, yerr=cv_stds, capsize=5,
                  color=['steelblue', 'orange', 'steelblue', 'green'], alpha=0.7,
                  error_kw={'linewidth': 2, 'ecolor': 'black'})

    # Add chance level line (1/6 = 16.7%)
    ax.axhline(y=1/6, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Chance (16.7%)')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Out-of-Sample Prediction Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # Add percentage labels
    for bar, mean in zip(bars, cv_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean*100:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'parameter_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def visualize_strategy_preferences(results: Dict, strategy_names: List[str]):
    """
    Create detailed visualization of strategy preference shifts across tasks.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    task_nums = [1, 2, 3, 4]
    task_labels = ['Task 1\n(No Restrictions)', 'Task 2\n(Deletion Only)',
                   'Task 3\n(No Restrictions)', 'Task 4\n(Time Pressure)']

    # Extract beta values
    beta_matrix = np.array([results[f'task_{i}']['params']['beta_strategies'] for i in task_nums])

    # 1. Grouped bar chart
    ax = axes[0]
    x = np.arange(len(strategy_names))
    width = 0.2

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, task_num in enumerate(task_nums):
        offset = (i - 1.5) * width
        ax.bar(x + offset, beta_matrix[i], width, label=task_labels[i],
               color=colors[i], alpha=0.8)

    ax.set_ylabel('β (Strategy Preference)', fontsize=13)
    ax.set_title('Strategy Preferences Across Tasks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in strategy_names],
                       rotation=45, ha='right', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # 2. Task-to-task differences
    ax = axes[1]

    # Calculate differences
    diff_1_2 = beta_matrix[1] - beta_matrix[0]  # Task 2 - Task 1 (constraint effect)
    diff_3_4 = beta_matrix[3] - beta_matrix[2]  # Task 4 - Task 3 (time pressure effect)

    x = np.arange(len(strategy_names))
    width = 0.35

    ax.bar(x - width/2, diff_1_2, width, label='Δ(Task 2 - Task 1)\nConstraint Effect',
           color='orange', alpha=0.8)
    ax.bar(x + width/2, diff_3_4, width, label='Δ(Task 4 - Task 3)\nTime Pressure Effect',
           color='green', alpha=0.8)

    ax.set_ylabel('Δβ (Change in Preference)', fontsize=13)
    ax.set_title('Strategy Preference Shifts', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in strategy_names],
                       rotation=45, ha='right', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'strategy_preferences.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def visualize_model_fit_quality(results: Dict):
    """
    Visualize cross-validation metrics to assess model fit quality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    task_nums = [1, 2, 3, 4]
    task_labels = ['Task 1\n(No Restrictions)', 'Task 2\n(Deletion Only)',
                   'Task 3\n(No Restrictions)', 'Task 4\n(Time Pressure)']

    # Extract CV metrics
    in_sample_loss = []
    out_sample_loss = []
    in_sample_loss_std = []
    out_sample_loss_std = []

    for i in task_nums:
        res = results[f'task_{i}']
        if 'cv' in res:
            in_sample_loss.append(res['cv']['in_sample_loss_mean'])
            out_sample_loss.append(res['cv']['out_sample_loss_mean'])
            in_sample_loss_std.append(res['cv']['in_sample_loss_std'])
            out_sample_loss_std.append(res['cv']['out_sample_loss_std'])

    x = np.arange(len(task_nums))
    width = 0.35

    # 1. Negative log-likelihood comparison
    ax = axes[0]

    ax.bar(x - width/2, in_sample_loss, width, yerr=in_sample_loss_std,
           capsize=5, label='In-Sample', color='steelblue', alpha=0.7,
           error_kw={'linewidth': 2, 'ecolor': 'black'})
    ax.bar(x + width/2, out_sample_loss, width, yerr=out_sample_loss_std,
           capsize=5, label='Out-of-Sample', color='coral', alpha=0.7,
           error_kw={'linewidth': 2, 'ecolor': 'black'})

    ax.set_ylabel('Negative Log-Likelihood', fontsize=13)
    ax.set_title('Model Fit: In-Sample vs Out-of-Sample', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 2. Overfitting analysis (out-sample - in-sample)
    ax = axes[1]

    overfitting = np.array(out_sample_loss) - np.array(in_sample_loss)
    colors_overfit = ['green' if o < 5 else 'orange' if o < 10 else 'red' for o in overfitting]

    bars = ax.bar(x, overfitting, width*1.5, color=colors_overfit, alpha=0.7,
                  edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Overfitting Gap\n(Out-Sample - In-Sample Neg-LL)', fontsize=12)
    ax.set_title('Overfitting Assessment', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add value labels
    for bar, val in zip(bars, overfitting):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=10, fontweight='bold')

    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Good (<5)'),
        Patch(facecolor='orange', alpha=0.7, label='Moderate (5-10)'),
        Patch(facecolor='red', alpha=0.7, label='High (>10)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, title='Overfitting Level')

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'model_fit_quality.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def save_model_results(results: Dict, strategy_names: List[str]):
    """
    Save model results to text file.
    """
    output_file = OUTPUT_DIR / 'model_results.txt'

    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COGNITIVE MODEL RESULTS\n")
        f.write("Password Editing Behavior Study\n")
        f.write("="*70 + "\n\n")

        # Individual task results
        for task_num in [1, 2, 3, 4]:
            task_key = f'task_{task_num}'
            res = results[task_key]
            params = res['params']

            f.write(f"TASK {task_num}\n")
            f.write("-"*70 + "\n")
            f.write(f"Data points: {res['n']}\n")
            f.write(f"Negative log-likelihood: {res['loss']:.2f}\n")

            # Add cross-validation results if available
            if 'cv' in res:
                cv = res['cv']
                f.write(f"\n5-Fold Cross-Validation:\n")
                f.write(f"  In-sample neg-LL:     {cv['in_sample_loss_mean']:.2f} ± {cv['in_sample_loss_std']:.2f}\n")
                f.write(f"  Out-of-sample neg-LL: {cv['out_sample_loss_mean']:.2f} ± {cv['out_sample_loss_std']:.2f}\n")
                f.write(f"  Out-of-sample accuracy: {cv['out_sample_acc_mean']:.3f} ± {cv['out_sample_acc_std']:.3f}\n")
            f.write("\n")

            f.write("Fitted Parameters:\n")
            f.write(f"  α_security:       {params['alpha_security']:.4f}\n")
            f.write(f"  α_memory:         {params['alpha_memory']:.4f}\n")
            f.write(f"  λ_constraint:     {params['lambda_constraint']:.4f}\n")
            f.write(f"  λ_time:           {params['lambda_time']:.4f}\n")
            f.write(f"  τ (rationality):  {params['tau']:.4f}\n\n")

            f.write("Strategy Preferences (β):\n")
            for i, name in enumerate(strategy_names):
                f.write(f"  {name:25s}: {params['beta_strategies'][i]:7.4f}\n")
            f.write("\n\n")

        # Comparisons
        f.write("="*70 + "\n")
        f.write("TASK COMPARISONS\n")
        f.write("="*70 + "\n\n")

        # Task 1 vs 2
        f.write("TASK 1 vs TASK 2 (Deletion Constraint Effect)\n")
        f.write("-"*70 + "\n")
        params_1 = results['task_1']['params']
        params_2 = results['task_2']['params']

        f.write("Parameter Changes (Task 2 - Task 1):\n")
        f.write(f"  Δα_security:      {params_2['alpha_security'] - params_1['alpha_security']:+7.4f}\n")
        f.write(f"  Δα_memory:        {params_2['alpha_memory'] - params_1['alpha_memory']:+7.4f}\n")
        f.write(f"  Δλ_constraint:    {params_2['lambda_constraint'] - params_1['lambda_constraint']:+7.4f}\n")
        f.write(f"  Δτ (rationality): {params_2['tau'] - params_1['tau']:+7.4f}\n\n")

        f.write("Strategy Preference Changes (Δβ):\n")
        for i, name in enumerate(strategy_names):
            delta = params_2['beta_strategies'][i] - params_1['beta_strategies'][i]
            f.write(f"  {name:25s}: {delta:+7.4f}\n")
        f.write("\n\n")

        # Task 3 vs 4
        f.write("TASK 3 vs TASK 4 (Time Pressure Effect)\n")
        f.write("-"*70 + "\n")
        params_3 = results['task_3']['params']
        params_4 = results['task_4']['params']

        f.write("Parameter Changes (Task 4 - Task 3):\n")
        f.write(f"  Δα_security:      {params_4['alpha_security'] - params_3['alpha_security']:+7.4f}\n")
        f.write(f"  Δα_memory:        {params_4['alpha_memory'] - params_3['alpha_memory']:+7.4f}\n")
        f.write(f"  Δλ_time:          {params_4['lambda_time'] - params_3['lambda_time']:+7.4f}\n")
        f.write(f"  Δτ (rationality): {params_4['tau'] - params_3['tau']:+7.4f}\n\n")

        f.write("Strategy Preference Changes (Δβ):\n")
        for i, name in enumerate(strategy_names):
            delta = params_4['beta_strategies'][i] - params_3['beta_strategies'][i]
            f.write(f"  {name:25s}: {delta:+7.4f}\n")
        f.write("\n")

    print(f"Saved: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run cognitive modeling pipeline.
    """
    # Load data
    df, data, strategy_to_idx, strategy_names = load_and_preprocess_data()

    # Count strategies for initialization
    strategy_counts = df['strategy_simplified'].value_counts().to_dict()

    # Initialize model
    print("\n--- Initializing Model ---")
    model = PasswordEditingModel(num_strategies=len(strategy_names))
    model.strategy_names = strategy_names

    # Initialize parameters
    initial_params = initialize_parameters(strategy_counts)

    # Fit models for each individual task
    results = fit_model_by_task(model, data, initial_params)

    # Analyze results
    analyze_fitted_parameters(results, strategy_names)

    # Visualize predictions
    print("\n--- Generating Visualizations ---")
    visualize_model_predictions(model, results, data, strategy_names)
    visualize_parameter_comparison(results, strategy_names)
    visualize_strategy_preferences(results, strategy_names)
    visualize_model_fit_quality(results)

    # Save results
    save_model_results(results, strategy_names)

    print("\n" + "="*70)
    print("MODEL FITTING COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {f.name}")
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)

    # Compare Task 1 vs 2 (deletion constraint)
    params_1 = results['task_1']['params']
    params_2 = results['task_2']['params']

    print(f"\nDELETION CONSTRAINT EFFECT (Task 1 → Task 2):")
    print(f"  Δλ_constraint: {params_2['lambda_constraint'] - params_1['lambda_constraint']:+.3f}")
    print(f"  Δτ (rationality): {params_2['tau'] - params_1['tau']:+.3f}")
    print(f"  Task 1 λ_constraint: {params_1['lambda_constraint']:.3f}")
    print(f"  Task 2 λ_constraint: {params_2['lambda_constraint']:.3f}")

    # Compare Task 3 vs 4 (time pressure)
    params_3 = results['task_3']['params']
    params_4 = results['task_4']['params']

    print(f"\nTIME PRESSURE EFFECT (Task 3 → Task 4):")
    print(f"  Δλ_time: {params_4['lambda_time'] - params_3['lambda_time']:+.3f}")
    print(f"  Δτ (rationality): {params_4['tau'] - params_3['tau']:+.3f}")
    print(f"  Task 3 τ: {params_3['tau']:.3f}")
    print(f"  Task 4 τ: {params_4['tau']:.3f}")

    print(f"\nSECURITY/MEMORY TRADEOFF RATIOS:")
    for task_num in [1, 2, 3, 4]:
        params = results[f'task_{task_num}']['params']
        ratio = params['alpha_security'] / params['alpha_memory']
        print(f"  Task {task_num}: {ratio:.2f}")

    print("\n")


if __name__ == "__main__":
    main()
