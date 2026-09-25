import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Financial Market Regime & Event Intelligence Engine\n",
    "## Notebook 08: Market Regime Explainability via Supervised SHAP Surrogate\n",
    "\n",
    "Welcome to Notebook 08! In this notebook, we introduce **Explainable AI (XAI)** to the Market Regime Engine by applying **SHAP (SHapley Additive exPlanations)** to interpret the quantitative boundaries defining HMM market regimes.\n",
    "\n",
    "### Objectives:\n",
    "1. **Recreate HMM Pipeline**: Download daily data (`^GSPC`, `TLT`, `GLD`), construct the 6-feature matrix, and decode 4 Gaussian HMM market regime states.\n",
    "2. **Train Supervised Surrogate Model**: Train a `RandomForestClassifier` to map the 6 market features to decoded HMM state labels using a chronological 80/20 train/test split.\n",
    "3. **Evaluate Surrogate Fidelity**: Measure how accurately the surrogate model reproduces HMM state assignments (Accuracy, Macro F1 for present classes, Macro F1 for all 4 states, Confusion Matrix).\n",
    "4. **Global SHAP Analysis Across Historical Regimes**: Calculate global SHAP feature importances across representative historical observations so all four HMM states are represented.\n",
    "5. **Local Observation SHAP Breakdown**: Explain individual test day regime predictions using local SHAP waterfall plots and feature contribution tables.\n",
    "6. **Financial Interpretation & Limitations**: Interpret feature impacts in economic terms and document model disclaimers."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Step 1: Conceptual Framework & Surrogate Approach\n",
    "\n",
    "#### Why a Surrogate Model is Required:\n",
    "- The **Gaussian Hidden Markov Model (HMM)** is an *unsupervised, sequential latent-variable model* governed by transition matrices and continuous multivariate emission distributions.\n",
    "- **SHAP (SHapley Additive exPlanations)** is designed for *supervised feature-to-label models* to calculate game-theoretic marginal feature contributions.\n",
    "- Direct application of SHAP to sequential unsupervised HMM transition mechanics is conceptually non-trivial and mathematically ill-defined.\n",
    "\n",
    "#### The Supervised Surrogate Workflow:\n",
    "1. We train a supervised decision tree ensemble (`RandomForestClassifier`) to predict the decoded HMM state ($s_t$) from the 6 quantitative market features ($X_t$).\n",
    "2. We evaluate **Surrogate Fidelity** — how closely $f_{\\text{surrogate}}(X_t)$ replicates the HMM state decodes $s_t$.\n",
    "3. We apply SHAP `TreeExplainer` to the surrogate model to reveal the precise feature thresholds and non-linear interactions driving regime assignments.\n",
    "\n",
    "#### Strict Conceptual Distinction:\n",
    "- **HMM State Detection**: Unsupervised sequential state decoding from multivariate financial emissions.\n",
    "- **Surrogate Model**: Supervised decision-tree classifier trained to mimic HMM state labels.\n",
    "- **SHAP Explanation**: Game-theoretic feature attribution explaining the surrogate model's decision rules."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Step 2: Import Required Libraries\n",
    "\n",
    "**Why we do this:**\n",
    "- `pandas` & `numpy`: Data handling and array manipulation.\n",
    "- `matplotlib.pyplot`: Custom visualization rendering.\n",
    "- `yfinance`: Downloading historical asset prices.\n",
    "- `sklearn.preprocessing.StandardScaler`: Standardizing features for HMM fitting.\n",
    "- `hmmlearn.hmm.GaussianHMM`: 4-state Gaussian HMM market regime model.\n",
    "- `sklearn.ensemble.RandomForestClassifier`: Supervised tree surrogate model.\n",
    "- `sklearn.metrics`: Evaluating surrogate fidelity (Accuracy, Macro F1, Confusion Matrix).\n",
    "- `shap`: Game-theoretic model explainability framework."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import yfinance as yf\n",
    "import shap\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from hmmlearn.hmm import GaussianHMM\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix\n",
    "\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_colwidth', 90)\n",
    "print(\"Libraries successfully imported!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Step 3: Recreate Market Data & 4-State Gaussian HMM Pipeline\n",
    "\n",
    "We download 5 years of daily market data (`^GSPC`, `TLT`, `GLD`), construct the exact 6 quantitative features, standardize inputs, and fit the 4-state Gaussian HMM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download historical closing prices\n",
    "sp500_close = yf.download(\"^GSPC\", period=\"5y\", interval=\"1d\")[\"Close\"]\n",
    "tlt_close = yf.download(\"TLT\", period=\"5y\", interval=\"1d\")[\"Close\"]\n",
    "gld_close = yf.download(\"GLD\", period=\"5y\", interval=\"1d\")[\"Close\"]\n",
    "\n",
    "if isinstance(sp500_close, pd.DataFrame): sp500_close = sp500_close.squeeze()\n",
    "if isinstance(tlt_close, pd.DataFrame): tlt_close = tlt_close.squeeze()\n",
    "if isinstance(gld_close, pd.DataFrame): gld_close = gld_close.squeeze()\n",
    "\n",
    "# Construct Quantitative Financial Features\n",
    "daily_return = sp500_close.pct_change()\n",
    "vol_20 = daily_return.rolling(window=20).std()\n",
    "mom_20 = sp500_close.pct_change(periods=20)\n",
    "peak = sp500_close.cummax()\n",
    "drawdown = (sp500_close - peak) / peak\n",
    "\n",
    "tlt_return = tlt_close.pct_change()\n",
    "gld_return = gld_close.pct_change()\n",
    "\n",
    "corr_sp_tlt = daily_return.rolling(window=20).corr(tlt_return)\n",
    "corr_sp_gld = daily_return.rolling(window=20).corr(gld_return)\n",
    "\n",
    "feature_names = [\n",
    "    \"Daily_Return\", \"Rolling_Volatility_20\", \"Momentum_20\",\n",
    "    \"Drawdown\", \"SP500_TLT_Corr_20\", \"SP500_GLD_Corr_20\"\n",
    "]\n",
    "\n",
    "market_raw_df = pd.DataFrame({\n",
    "    \"Close\": sp500_close,\n",
    "    \"Daily_Return\": daily_return,\n",
    "    \"Rolling_Volatility_20\": vol_20,\n",
    "    \"Momentum_20\": mom_20,\n",
    "    \"Drawdown\": drawdown,\n",
    "    \"SP500_TLT_Corr_20\": corr_sp_tlt,\n",
    "    \"SP500_GLD_Corr_20\": corr_sp_gld\n",
    "})\n",
    "\n",
    "# Remove incomplete rolling rows\n",
    "market_clean_df = market_raw_df.dropna(subset=feature_names).copy()\n",
    "\n",
    "# Standardize features & fit 4-state Gaussian HMM\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(market_clean_df[feature_names])\n",
    "\n",
    "hmm_model = GaussianHMM(n_components=4, covariance_type=\"full\", n_iter=200, random_state=42)\n",
    "hmm_model.fit(X_scaled)\n",
    "market_clean_df[\"HMM_State\"] = hmm_model.predict(X_scaled)\n",
    "\n",
    "print(f\"Market & HMM Data ready. Total observations: {len(market_clean_df)}\")\n",
    "print(\"Decoded HMM State Frequencies:\")\n",
    "print(market_clean_df[\"HMM_State\"].value_counts().sort_index())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Step 4: Chronological Train/Test Split & Surrogate Model Training\n",
    "\n",
    "We split the dataset **chronologically** (80% training set, 20% test set) without shuffling to strictly respect time series sequence. We fit a `RandomForestClassifier` as our surrogate explainer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = market_clean_df[feature_names]\n",
    "y = market_clean_df[\"HMM_State\"]\n",
    "\n",
    "# Chronological Train/Test Split (80% Train, 20% Test)\n",
    "train_size = int(len(X) * 0.8)\n",
    "X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]\n",
    "y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]\n",
    "\n",
    "print(f\"Chronological Split: Training samples = {len(X_train)} ({X_train.index.min().strftime('%Y-%m-%d')} to {X_train.index.max().strftime('%Y-%m-%d')})\")\n",
    "print(f\"Chronological Split: Testing samples  = {len(X_test)} ({X_test.index.min().strftime('%Y-%m-%d')} to {X_test.index.max().strftime('%Y-%m-%d')})\")\n",
    "\n",
    "# Train Random Forest Surrogate Model\n",
    "rf_surrogate = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=\"balanced\")\n",
    "rf_surrogate.fit(X_train, y_train)\n",
    "print(\"Random Forest Surrogate Model trained successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Step 5: Evaluate Primary Out-of-Sample Surrogate Fidelity\n",
    "\n",
    "#### What is Surrogate Fidelity?\n",
    "- **Surrogate Fidelity** measures how accurately the Random Forest surrogate reproduces the unsupervised HMM's state assignments on out-of-sample test data.\n",
    "- High surrogate fidelity validates that SHAP feature attributions on the Random Forest accurately reflect the quantitative decision boundaries of the HMM.\n",
    "- *Note*: This evaluates model approximation quality, **NOT** predictive accuracy of future asset prices or returns.\n",
    "\n",
    "> **Test Period State Distribution Disclaimer**:\n",
    "> The chronological test period contains only HMM States 0 and 3. Therefore, out-of-sample surrogate fidelity can only be directly evaluated for those states. Metrics involving all four labels assign zero support to States 1 and 2 and should not be interpreted as evidence of poor classification of those states."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = rf_surrogate.predict(X_test)\n",
    "\n",
    "surrogate_acc = accuracy_score(y_test, y_pred)\n",
    "present_classes = np.unique(y_test)\n",
    "macro_f1_present = f1_score(y_test, y_pred, labels=present_classes, average=\"macro\")\n",
    "macro_f1_all = f1_score(y_test, y_pred, labels=[0, 1, 2, 3], average=\"macro\", zero_division=0)\n",
    "all_states = [0, 1, 2, 3]\n",
    "surrogate_cm = confusion_matrix(y_test, y_pred, labels=all_states)\n",
    "\n",
    "print(\"=== SURROGATE MODEL FIDELITY EVALUATION ===\")\n",
    "print(f\"Primary Test Accuracy                   : {surrogate_acc:.4f}\")\n",
    "print(f\"Macro F1 (Classes Present in Test {present_classes.tolist()}): {macro_f1_present:.4f}\")\n",
    "print(f\"Macro F1 (All 4 HMM States [0, 1, 2, 3])   : {macro_f1_all:.4f}  (Note: States 1 & 2 have 0 support in test period)\\n\")\n",
    "\n",
    "print(\"--- Test Set State Counts ---\")\n",
    "test_state_counts = pd.Series(y_test).value_counts().reindex(all_states, fill_value=0)\n",
    "for s, cnt in test_state_counts.items():\n",
    "    print(f\"  HMM State {s}: {cnt:>3} samples\")\n",
    "\n",
    "print(\"\\n--- Full Classification Report (All 4 States) ---\")\n",
    "print(classification_report(y_test, y_pred, labels=all_states, zero_division=0))\n",
    "\n",
    "print(\"--- Confusion Matrix (All 4 HMM States) ---\")\n",
    "cm_df = pd.DataFrame(\n",
    "    surrogate_cm,\n",
    "    index=[f\"Actual HMM State {i}\" for i in all_states],\n",
    "    columns=[f\"Pred Surrogate State {i}\" for i in all_states]\n",
    ")\n",
    "display(cm_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Step 6: Global SHAP Analysis Across Historical Regimes\n",
    "\n",
    "To ensure global feature importance is not restricted to the 2025–2026 low-volatility test window, we use the trained Random Forest surrogate model to compute SHAP values across representative historical observations covering the full 5-year dataset ($N=1,235$), ensuring all four HMM states are represented.\n",
    "\n",
    "*Note: The surrogate model remains trained strictly on the training split $X_{\\text{train}}$ and is NOT retrained on future data.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize SHAP TreeExplainer on trained surrogate model\n",
    "explainer = shap.TreeExplainer(rf_surrogate)\n",
    "\n",
    "# Compute SHAP values across full historical dataset X so all 4 states are represented\n",
    "shap_full = explainer(X)\n",
    "\n",
    "# 1. Global Mean Absolute SHAP Feature Importance Table across full historical regimes\n",
    "mean_abs_shap_full = np.abs(shap_full.values).mean(axis=(0, 2))\n",
    "global_shap_full_df = pd.DataFrame({\n",
    "    \"Feature\": feature_names,\n",
    "    \"Mean_Absolute_SHAP\": mean_abs_shap_full\n",
    "}).sort_values(by=\"Mean_Absolute_SHAP\", ascending=False).reset_index(drop=True)\n",
    "\n",
    "print(\"--- Global SHAP Feature Importance Table (Full Historical Regimes) ---\")\n",
    "display(global_shap_full_df.round(4))\n",
    "\n",
    "# 2. SHAP Summary Beeswarm Plot across full historical period (State 0 representative)\n",
    "plt.figure(figsize=(10, 6))\n",
    "shap.plots.beeswarm(shap_full[:, :, 0], show=False)\n",
    "plt.title(\"SHAP Beeswarm Summary Plot (Full Historical Regimes - State 0)\", fontsize=12, fontweight=\"bold\")\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# 3. SHAP Feature Importance Bar Plot\n",
    "plt.figure(figsize=(9, 5))\n",
    "shap.plots.bar(shap_full[:, :, 0], show=False)\n",
    "plt.title(\"Global SHAP Feature Importance Bar Plot (Full Historical Regimes)\", fontsize=12, fontweight=\"bold\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Step 7: Local Observation SHAP Explanation\n",
    "\n",
    "We select an individual observation from the out-of-sample test set (`2025-09-25`) to inspect local SHAP feature attributions using a Waterfall Plot."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_idx = 0\n",
    "sample_date = X_test.index[sample_idx].strftime(\"%Y-%m-%d\")\n",
    "actual_hmm_state = int(y_test.iloc[sample_idx])\n",
    "pred_surrogate_state = int(y_pred[sample_idx])\n",
    "is_correct = (actual_hmm_state == pred_surrogate_state)\n",
    "sample_features = X_test.iloc[sample_idx]\n",
    "\n",
    "print(f\"=== Local Observation Inspection ({sample_date}) ===\")\n",
    "print(f\"Observation Date           : {sample_date}\")\n",
    "print(f\"Actual HMM Decoded State   : State {actual_hmm_state}\")\n",
    "print(f\"Surrogate Predicted State : State {pred_surrogate_state}\")\n",
    "print(f\"Correct Prediction        : {is_correct}\")\n",
    "print(\"\\n--- Feature Values for Observation ---\")\n",
    "for feat, val in sample_features.items():\n",
    "    print(f\"  {feat:<22}: {val:+.6f}\")\n",
    "\n",
    "# Compute test set SHAP for waterfall plot\n",
    "shap_test = explainer(X_test)\n",
    "\n",
    "# Plot Waterfall for this observation for the predicted state\n",
    "plt.figure(figsize=(10, 6))\n",
    "shap.plots.waterfall(shap_test[sample_idx, :, pred_surrogate_state], show=False)\n",
    "plt.title(f\"Local SHAP Waterfall Explanation for {sample_date} (Predicted State {pred_surrogate_state}, Match={is_correct})\", fontsize=12, fontweight=\"bold\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Step 8: Financial Interpretation of SHAP Explainability\n",
    "\n",
    "1. **`Drawdown` Has the Highest Global Contribution**:\n",
    "   - `Drawdown` is the single most decisive feature defining market regimes. Severe cumulative drawdowns distinctly demarcate High-Volatility Bear (State 1) and Volatile Recovery (State 2) regimes from Low-Volatility Bull (State 0) regimes.\n",
    "\n",
    "2. **`Momentum_20` is the Second-Highest Driver**:\n",
    "   - 20-day price momentum provides crucial directional context, separating trending recovery regimes from persistent downward sell-offs.\n",
    "\n",
    "3. **Volatility & Cross-Asset Correlations Provide Additional Regime Separation**:\n",
    "   - `Rolling_Volatility_20` acts as a sharp threshold for turbulent vs. calm regimes.\n",
    "   - Cross-asset correlations (`SP500_TLT_Corr_20` & `SP500_GLD_Corr_20`) provide macroeconomic context regarding Treasury flight-to-safety and commodity hedging behaviors.\n",
    "\n",
    "4. **`Daily_Return` Has Relatively Low Global Contribution**:\n",
    "   - Daily single-session return has low global SHAP contribution because individual daily price noise lacks the cumulative multi-week signal required to define persistent market regimes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### Step 9: Limitations of Surrogate SHAP Explainability\n",
    "\n",
    "1. **Surrogate Approximation Limit**: SHAP explains the Random Forest surrogate's approximation of HMM state assignments, not the internal HMM transition/emission mechanism.\n",
    "2. **Latent Model-Derived Labels**: HMM states are statistical latent clusters derived from quantitative market data rather than directly observable physical states.\n",
    "3. **Multicollinearity Impact**: Financial features (e.g., Drawdown, Volatility, Momentum) exhibit correlation, which can distribute SHAP values across co-linear variables.\n",
    "4. **Fidelity Dependency**: The validity of the SHAP explanation relies on maintaining high surrogate fidelity.\n",
    "5. **No Trading Advice**: This analysis is strictly historical and observational. It does not constitute trading signals or financial advice."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Final Summary Output Print/Display Report\n",
    "print(\"=========================================================\")\n",
    "print(\"   EXPLAINABLE HMM REGIME SURROGATE SUMMARY REPORT\")\n",
    "print(\"=========================================================\")\n",
    "print(f\"1. Primary Test Accuracy                : {surrogate_acc:.4f}\")\n",
    "print(f\"2. Macro F1 (Classes Present in Test [0, 3]): {macro_f1_present:.4f}\")\n",
    "print(f\"3. Macro F1 (Across All 4 States [0, 1, 2, 3]): {macro_f1_all:.4f}  (Note: States 1 & 2 have 0 support in test period)\")\n",
    "print(\"\\n4. Test Set State Counts:\")\n",
    "for s, cnt in test_state_counts.items():\n",
    "    print(f\"   State {s}: {cnt:>3} samples\")\n",
    "print(\"\\n5. Global SHAP Feature Importance Ranking (Full Historical Regimes):\")\n",
    "for rank, row in global_shap_full_df.iterrows():\n",
    "    print(f\"   Rank {rank+1}: {row['Feature']:<22} (Mean |SHAP| = {row['Mean_Absolute_SHAP']:.6f})\")\n",
    "print(f\"\\n6. Local SHAP Explanation Sample ({sample_date}):\")\n",
    "print(f\"   Actual HMM State    : State {actual_hmm_state}\")\n",
    "print(f\"   Predicted Surrogate : State {pred_surrogate_state}\")\n",
    "print(f\"   Prediction Match    : {is_correct}\")\n",
    "print(\"=========================================================\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("notebooks/08_regime_explainability_shap.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print("notebooks/08_regime_explainability_shap.ipynb successfully updated!")
