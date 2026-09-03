import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PIL import Image
from data_prep import load_and_preprocess_data

class FraudExplainer:
    """Explains cyber fraud predictions using clean, non-overlapping SHAP metrics."""

    def __init__(
        self,
        # FIXED BUG 1: Points to Suyash's new XGBoost model
        model_path: str = os.path.join("models", "new_fraud_model.pkl"),
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


if __name__ == "__main__":
    print("Executing SHAP Explainer Module...")
    
    # FIXED BUG 2: Using the real data pipeline from Ekansh instead of fake columns
    _, X_test, _, _, _ = load_and_preprocess_data()
    
    explainer = FraudExplainer()
    
    # Find the most severe fraud case in the test set to demonstrate
    probabilities = explainer.model.predict_proba(X_test)[:, 1]
    highest_risk_index = probabilities.argmax()
    
    sample_transaction = X_test.iloc[[highest_risk_index]]
    background_data = X_test.sample(100, random_state=42) # 100 real background samples
    
    explainer.generate_audit_dashboard(
        "TXN_CRITICAL_001", sample_transaction, background_data
    )