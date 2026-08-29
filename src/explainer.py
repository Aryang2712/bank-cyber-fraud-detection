import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class FraudExplainer:
    """Explains cyber fraud predictions using clean 4-panel SHAP metrics."""

    def __init__(
        self,
        model_path: str = os.path.join("models", "random_forest.pkl"),
        scaler_path: str = os.path.join("models", "scaler.pkl"),
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Run src/train.py first."
            )
        self.model = joblib.load(model_path)
        self.scaler = (
            joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        )
        self.explainer = shap.Explainer(self.model)

    def generate_audit_dashboard(
        self,
        transaction_id: str,
        transaction_df: pd.DataFrame,
        background_data: pd.DataFrame,
        output_path: str = os.path.join(
            "reports", "figures", "advanced_shap_audit.png"
        ),
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        shap_values_sample = self.explainer(transaction_df)
        shap_values_global = self.explainer(background_data)

        # Create large figure with proper spacing
        fig = plt.figure(figsize=(20, 12))

        # Panel A: Summary Plot (Top-Left)
        ax1 = fig.add_subplot(2, 2, 1)
        plt.sca(ax1)
        shap.summary_plot(
            shap_values_global[:, :, 1]
            if len(shap_values_global.shape) == 3
            else shap_values_global,
            background_data,
            show=False,
            plot_type="dot",
        )
        ax1.set_title(
            "Global Feature Impact (SHAP Summary)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        # Panel B: Multi-line Decision Plot (Top-Right)
        ax2 = fig.add_subplot(2, 2, 2)
        plt.sca(ax2)
        bg_shap_sample = (
            shap_values_global.values[:10, :, 1]
            if len(shap_values_global.shape) == 3
            else shap_values_global.values[:10]
        )
        base_val = (
            self.explainer.expected_value[1]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value
        )

        shap.decision_plot(
            base_val,
            bg_shap_sample,
            background_data.iloc[:10],
            feature_names=background_data.columns.tolist(),
            show=False,
        )
        ax2.set_title(
            f"Interactive Decision Convergence ({transaction_id})",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        # Panel C: Feature Importance Bar Chart (Bottom-Left)
        ax3 = fig.add_subplot(2, 2, 3)
        plt.sca(ax3)
        importance_scores = pd.DataFrame(
            {
                "Feature": background_data.columns,
                "Importance": self.model.feature_importances_,
            }
        ).sort_values(by="Importance", ascending=True)

        ax3.barh(
            importance_scores["Feature"],
            importance_scores["Importance"],
            color="#1f77b4",
        )
        ax3.set_title(
            "Model-Wide Feature Importance",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax3.set_xlabel("Relative Importance Score", fontsize=10)

        # Panel D: Local Explanation Heatmap (Bottom-Right)
        ax4 = fig.add_subplot(2, 2, 4)
        plt.sca(ax4)
        shap_values_heatmap = (
            shap_values_global[:20, :, 1]
            if len(shap_values_global.shape) == 3
            else shap_values_global[:20]
        )
        shap.plots.heatmap(shap_values_heatmap, show=False)
        ax4.set_title(
            "Local Explanation Clusters (Heatmap)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        # Main Title & Subplot Spacing (prevents text overlap)
        fig.suptitle(
            "XAI Audit Deep-Dive: Shadow Transfer Risk Profiling",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        plt.subplots_adjust(
            left=0.1, right=0.95, top=0.90, bottom=0.08, wspace=0.35, hspace=0.35
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(
            f"[SUCCESS] Clean 4-panel SHAP audit plot saved to: {output_path}"
        )


def generate_digital_arrest_sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Active_Call_Duration_Min": 120.0,
                "Transaction_Amount": 450000.0,
                "OTP_Failed_Attempts": 3.0,
                "New_Payee_Added_Mins_Ago": 5.0,
                "Is_New_Device": 1.0,
                "Is_High_Risk_IP": 1.0,
                "Account_Age_Days": 15.0,
            }
        ]
    )


def generate_background_data(n_samples=200) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Active_Call_Duration_Min": np.random.uniform(
                0.5, 120.0, n_samples
            ),
            "Transaction_Amount": np.random.uniform(10.0, 500000.0, n_samples),
            "OTP_Failed_Attempts": np.random.randint(0, 5, n_samples),
            "New_Payee_Added_Mins_Ago": np.random.uniform(
                1.0, 10000.0, n_samples
            ),
            "Is_New_Device": np.random.binomial(1, 0.1, n_samples),
            "Is_High_Risk_IP": np.random.binomial(1, 0.05, n_samples),
            "Account_Age_Days": np.random.uniform(1.0, 5000.0, n_samples),
        }
    )


if __name__ == "__main__":
    print("Executing SHAP Explainer Module...")
    sample_transaction = generate_digital_arrest_sample()
    background_data = generate_background_data()

    explainer = FraudExplainer()
    explainer.generate_audit_dashboard(
        "T12345", sample_transaction, background_data
    )