"""
Run linear regression diagnostics for hazelnut biomass vs. canopy volume.

This module provides a CLI wrapper for regression analysis used to evaluate
biomass–volume relationships. It performs the following tasks:
1. Load a CSV dataset containing 'biomass_kg' and 'volume_m3' columns.
2. Fit raw and log-transformed linear regression models.
3. Compute R², residual normality (Shapiro-Wilk), and approximate confidence intervals.
4. Fit a pooled regression model across all data and report absolute and relative RMSE.
5. Optionally display a diagnostic plot of observed vs. fitted biomass.
"""

import argparse
import sys
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import shapiro, t


def load_data(csv_path):
    """
    Load and clean input CSV data.

    Args:
        csv_path (str): Path to CSV file with biomass and volume data.

    Returns:
        tuple: (df_full, df)
            - df_full: Full dataframe with missing rows dropped.
            - df: Copy of df_full without 'site' and 'treeID' columns (if present).

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df_full = pd.read_csv(csv_path).dropna()
    df = df_full.copy()
    if 'site' in df.columns and 'treeID' in df.columns:
        df = df.drop(columns=['site', 'treeID'])
    return df_full, df


def run_regressions(df_full, df, do_print=True):
    """
    Fit raw and log-transformed linear regressions and compute diagnostics.

    Args:
        df_full (DataFrame): Full dataset (used for pooled model evaluation).
        df (DataFrame): Cleaned dataset for fitting regression models.
        do_print (bool, optional): If True, prints model summaries.

    Returns:
        dict: Results containing fitted models and diagnostics:
            - 'lr_raw': Linear model on raw biomass.
            - 'lr_log': Linear model on log-transformed biomass.
            - 'lr_pooled': Pooled linear model.
            - 'pooled_r2': R² of pooled model.
            - 'pooled_rmse': RMSE of pooled model.
            - 'y_pred': Predicted biomass values from pooled model.
    """
    X = df[['volume_m3']]
    y = df['biomass_kg']
    y_log = np.log1p(y)

    lr_raw = LinearRegression(fit_intercept=False).fit(X, y)
    lr_log = LinearRegression(fit_intercept=False).fit(X, y_log)

    resid_raw = y - lr_raw.predict(X)
    resid_log = y_log - lr_log.predict(X)

    if do_print:
        print("Model fit:")
        try:
            sh_raw_p = shapiro(resid_raw).pvalue
        except Exception:
            sh_raw_p = float('nan')
        try:
            sh_log_p = shapiro(resid_log).pvalue
        except Exception:
            sh_log_p = float('nan')
        print(f"  Raw: R² = {r2_score(y, lr_raw.predict(X)):.3f}, Shapiro p = {sh_raw_p:.3f}")
        print(f"  Log: R² = {r2_score(y_log, lr_log.predict(X)):.3f}, Shapiro p = {sh_log_p:.3f}")

    n = len(y)
    dof = max(1, n - 1)
    mse = np.sum((y - lr_raw.predict(X)) ** 2) / dof
    se = np.sqrt(mse)
    t_val = t.ppf(0.975, dof)
    ci = t_val * se
    if do_print:
        print(f"Raw model: biomass = {lr_raw.coef_[0]:.3f} × volume ± {ci:.1f} kg (approx 95% CI)")

    lr_pooled = LinearRegression(fit_intercept=False).fit(df_full[['volume_m3']], df_full['biomass_kg'])
    y_pred = lr_pooled.predict(df_full[['volume_m3']])
    r2 = r2_score(df_full['biomass_kg'], y_pred)
    rmse = np.sqrt(mean_squared_error(df_full['biomass_kg'], y_pred))
    rel_rmse = rmse / df_full['biomass_kg'].mean()
    if do_print:
        print(f"Pooled: coef = {lr_pooled.coef_[0]:.3f}, R² = {r2:.3f}, RMSE = {rmse:.1f}, rel_rmse = {rel_rmse:.3f}, n = {len(df_full)}")

    out = {
        'lr_raw': lr_raw,
        'lr_log': lr_log,
        'lr_pooled': lr_pooled,
        'pooled_r2': r2,
        'pooled_rmse': rmse,
        'y_pred': y_pred
    }
    return out


def optional_plot(df_full, y_pred):
    """
    Plot observed vs. fitted biomass if matplotlib is available.

    Args:
        df_full (DataFrame): Full dataset with observed biomass and volume.
        y_pred (np.ndarray): Predicted biomass from pooled model.

    Returns:
        None
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        warnings.warn("matplotlib not available; skipping plots")
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(df_full['volume_m3'], df_full['biomass_kg'], alpha=0.5, label='Observed')
    plt.plot(df_full['volume_m3'], y_pred, color='red', label='Fitted')
    plt.xlabel('Volume (m³)')
    plt.ylabel('Biomass (kg)')
    plt.title('Pooled Linear Regression: Biomass vs Volume')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main(argv=None):
    """
    Command-line entry point for linear regression diagnostics.

    Args:
        argv (list, optional): Command-line arguments.

    Returns:
        int: Exit code (0 = success, 2 = data load error).
    """
    parser = argparse.ArgumentParser(description='Linear regression diagnostics for hazelnut biomass')
    parser.add_argument('--csv', required=True, help='Path to CSV with biomass and volume columns')
    parser.add_argument('--plot', action='store_true', help='Show basic diagnostic plot (requires matplotlib)')
    args = parser.parse_args(argv)

    try:
        df_full, df = load_data(args.csv)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 2

    results = run_regressions(df_full, df, do_print=True)
    if args.plot:
        optional_plot(df_full, results['y_pred'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
