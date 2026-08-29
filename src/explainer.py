import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PIL import Image


class FraudExplainer:
    """Explains cyber fraud predictions using clean, non-overlapping SHAP metrics."""

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
        temp_dir = os.path.join("reports", "figures", "temp")
        os.makedirs(temp_dir, exist_ok=True)

        shap_values_sample = self.explainer(transaction_df)
        shap_values_global = self.explainer(background_data)

        # ------------------ Plot 1: Summary Plot ------------------
        plt.figure(figsize=(7, 5))
        shap.summary_plot(
            shap_values_global[:, :, 1]
            if len(shap_values_global.shape) == 3
            else shap_values_global,
            background_data,
            show=False,
            plot_type="dot",
        )
        plt.title("Global Feature Impact (SHAP Summary)", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        path1 = os.path.join(temp_dir, "plot1.png")
        plt.savefig(path1, dpi=200, bbox_inches="tight")
        plt.close("all")

        # ------------------ Plot 2: Decision Plot ------------------
        plt.figure(figsize=(7, 5))
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
        plt.title(f"Decision Convergence ({transaction_id})", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        path2 = os.path.join(temp_dir, "plot2.png")
        plt.savefig(path2, dpi=200, bbox_inches="tight")
        plt.close("all")

        # ------------------ Plot 3: Importance Bar ------------------
        plt.figure(figsize=(7, 5))
        importance_scores = pd.DataFrame(
            {
                "Feature": background_data.columns,
                "Importance": self.model.feature_importances_,
            }
        ).sort_values(by="Importance", ascending=True)

        plt.barh(importance_scores["Feature"], importance_scores["Importance"], color="#1f77b4")
        plt.title("Model-Wide Feature Importance", fontsize=11, fontweight="bold", pad=12)
        plt.xlabel("Relative Importance Score", fontsize=10)
        plt.tight_layout()
        path3 = os.path.join(temp_dir, "plot3.png")
        plt.savefig(path3, dpi=200, bbox_inches="tight")
        plt.close("all")

        # ------------------ Plot 4: Local Heatmap ------------------
        plt.figure(figsize=(7, 5))
        shap_values_heatmap = (
            shap_values_global[:20, :, 1]
            if len(shap_values_global.shape) == 3
            else shap_values_global[:20]
        )
        shap.plots.heatmap(shap_values_heatmap, show=False)
        plt.title("Local Explanation Clusters (Heatmap)", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        path4 = os.path.join(temp_dir, "plot4.png")
        plt.savefig(path4, dpi=200, bbox_inches="tight")
        plt.close("all")

        # ------------------ Stitch Plots into Final Canvas ------------------
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        img3 = Image.open(path3)
        img4 = Image.open(path4)

        # Target size for each quadrant
        w, h = 1200, 900
        img1 = img1.resize((w, h), Image.Resampling.LANCZOS)
        img2 = img2.resize((w, h), Image.Resampling.LANCZOS)
        img3 = img3.resize((w, h), Image.Resampling.LANCZOS)
        img4 = img4.resize((w, h), Image.Resampling.LANCZOS)

        title_height = 120
        canvas_w = w * 2 + 60
        canvas_h = h * 2 + title_height + 60

        # White canvas background
        dashboard = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

        # Paste subplots
        dashboard.paste(img1, (20, title_height))
        dashboard.paste(img2, (w + 40, title_height))
        dashboard.paste(img3, (20, title_height + h + 20))
        dashboard.paste(img4, (w + 40, title_height + h + 20))

        # Add global super-title directly via Matplotlib canvas overlay
        fig, ax = plt.subplots(figsize=(canvas_w / 100, canvas_h / 100), dpi=100)
        ax.imshow(dashboard)
        ax.axis("off")
        plt.title(
            "XAI Audit Deep-Dive: Shadow Transfer Risk Profiling",
            fontsize=22,
            fontweight="bold",
            pad=20,
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        # Cleanup temporary slice files
        for p in [path1, path2, path3, path4]:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

        print(f"[SUCCESS] Clean audit dashboard saved to: {output_path}")


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
            "Active_Call_Duration_Min": np.random.uniform(0.5, 120.0, n_samples),
            "Transaction_Amount": np.random.uniform(10.0, 500000.0, n_samples),
            "OTP_Failed_Attempts": np.random.randint(0, 5, n_samples),
            "New_Payee_Added_Mins_Ago": np.random.uniform(1.0, 10000.0, n_samples),
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