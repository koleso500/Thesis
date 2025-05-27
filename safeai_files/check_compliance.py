import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

from safeai_files.check_explainability import compute_rge_values
from safeai_files.check_robustness import rgr_all
from safeai_files.core import partial_rga_with_curves, rga


def compliance_curves(x_train, x_test, y_test, y_prob, model):
    """
    Compute SafeAI lists of values for plotting curves: Accuracy (RGA), Explainability (RGE AUC), Robustness (RGR AUC).

    Parameters:
    -------------
    x_train: pandas.DataFrame
        Training data features.
    x_test: pandas.DataFrame
        Test data features.
    y_test: pd.DataFrame
        True labels for test data.
    y_prob: list
        Predicted probabilities for the positive class.
    model: Union[CatBoostClassifier, CatBoostRegressor, XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
        Trained classifier used in compute_rge_values and rgr_all.

    Returns:
    --------
    dict containing:
        rga_value: float
        rge_auc: float
        rgr_auc: float
        x_final: list of float
        y_final: list of float
        z_final: list of float
    """

    # Accuracy (RGA)
    rga_value = rga(y_test, y_prob)

    # Explainability (RGE)
    explain = x_train.columns.tolist()
    remaining_vars = explain.copy()
    removed_vars = []
    step_rges = []

    for k in range(0, len(explain) + 1):
        if k == 0:
            step_rges.append(0.0)
            continue

        candidate_rges = []
        for var in remaining_vars:
            current_vars = removed_vars + [var]
            rge_k = compute_rge_values(x_train, x_test, y_prob, model, current_vars, group=True)
            candidate_rges.append((var, rge_k.iloc[0, 0]))

        best_var, best_rge = max(candidate_rges, key=lambda x: x[1])
        removed_vars.append(best_var)
        remaining_vars.remove(best_var)
        step_rges.append(best_rge)

    x_rge = np.linspace(0, 1, len(step_rges))
    y_rge = np.array(step_rges)
    rge_auc = auc(x_rge, y_rge)

    # Plot
    model_name = model.__class__.__name__
    plt.figure(figsize=(6, 4))
    plt.plot(x_rge, y_rge, marker='o', label=f"RGE Curve (AURGE = {rge_auc:.4f})")
    # Plot baseline only if not a Dummy model
    if model_name not in ["DummyRegressor", "DummyClassifier"]:
        random_baseline = float(y_rge[-1])
        plt.axhline(random_baseline, color='red', linestyle='--',
                    label=f"Random Baseline (RGE = {random_baseline:.2f})")
    plt.xlabel("Fraction of Variables Removed")
    plt.ylabel("RGE")
    plt.title(f"{model_name} RGE Curve")
    plt.legend()
    plt.grid(True)

    # Robustness (RGR)
    thresholds = np.arange(0, 0.51, 0.01)
    rgr_scores = [rgr_all(x_test, y_prob, model, t) for t in thresholds]
    normalized_t = thresholds / 0.5
    rgr_auc = auc(normalized_t, rgr_scores)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(normalized_t, rgr_scores, linestyle='-', label=f"RGR Curve (AURGR = {rgr_auc:.4f})")
    plt.title(f'{model_name} RGR Curve')
    if model_name not in ["DummyRegressor", "DummyClassifier"]:
        plt.axhline(0.5, color='red', linestyle='--', label=f"Random Baseline (RGE = 0.5)")
    plt.xlabel('Normalized Perturbation')
    plt.ylabel('RGR')
    plt.legend()
    plt.xlim([0, 1])
    plt.grid(True)

    # Values for final compliance score
    # RGA
    num_steps = len(step_rges) - 1
    step_rgas = []
    thresholds_rga = np.linspace(1, 0, num_steps + 1)
    for i in range(num_steps):
        lower = float(thresholds_rga[i + 1])
        upper = float(thresholds_rga[i])
        partial = partial_rga_with_curves(y_test, y_prob, lower, upper, False)
        step_rgas.append(partial)
    reverse_cumulative = np.cumsum(step_rgas[::-1])[::-1]
    x_final = np.concatenate((reverse_cumulative, [0.])).tolist()

    # RGE
    y_final = step_rges

    # RGR
    num_steps_rgr = len(step_rges)
    thresholds_rgr = np.linspace(0, 0.5, num_steps_rgr)
    z_final = [rgr_all(x_test, y_prob, model, t) for t in thresholds_rgr]

    # Plot all graphs
    plt.show()

    return {
        'model_name': model.__class__.__name__,
        'rga_value': rga_value,
        'rge_auc': rge_auc,
        'rgr_auc': rgr_auc,
        'x_final': x_final,
        'y_final': y_final,
        'z_final': z_final
    }


