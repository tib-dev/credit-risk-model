import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc

sns.set_theme(style="whitegrid", context="talk")


def plot_training_dashboard(results_df, trained_models, X_test, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle(
        "Credit Risk Model Training Dashboard",
        fontsize=22,
        fontweight="bold"
    )

    # -------------------------------
    # 1. ROC-AUC Comparison
    # -------------------------------
    roc_sorted = results_df.sort_values("roc_auc", ascending=True)

    sns.barplot(
        data=roc_sorted,
        x="roc_auc",
        y="model",
        hue="model",          # <-- FIX
        palette="viridis",
        legend=False,         # <-- FIX
        ax=axes[0, 0]
    )

    axes[0, 0].set_title("ROC-AUC by Model", fontweight="bold")
    axes[0, 0].set_xlim(0, 1)

    for i, v in enumerate(roc_sorted["roc_auc"]):
        axes[0, 0].text(v + 0.01, i, f"{v:.3f}", va="center")

    # -------------------------------
    # 2. ROC Curves
    # -------------------------------
    roc_plotted = False

    for name, model in trained_models.items():
        if not hasattr(model, "predict_proba"):
            continue

        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        axes[0, 1].plot(
            fpr,
            tpr,
            lw=2,
            label=f"{name} (AUC={auc(fpr, tpr):.3f})"
        )
        roc_plotted = True

    axes[0, 1].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0, 1].set_title("ROC Curve Comparison", fontweight="bold")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")

    if roc_plotted:  # <-- FIX
        axes[0, 1].legend(loc="lower right")

    # -------------------------------
    # 3. F1 Score Comparison
    # -------------------------------
    f1_sorted = results_df.sort_values("f1_score", ascending=True)

    sns.barplot(
        data=f1_sorted,
        x="f1_score",
        y="model",
        hue="model",          # <-- FIX
        palette="magma",
        legend=False,         # <-- FIX
        ax=axes[1, 0]
    )

    axes[1, 0].set_title(
        "F1 Score (Precision–Recall Balance)", fontweight="bold")
    axes[1, 0].set_xlim(0, 1)

    for i, v in enumerate(f1_sorted["f1_score"]):
        axes[1, 0].text(v + 0.01, i, f"{v:.3f}", va="center")

    # -------------------------------
    # 4. Feature Importance (Best Model)
    # -------------------------------
    best_model_name = (
        results_df.sort_values("roc_auc", ascending=False)
        .iloc[0]["model"]
    )
    best_model = trained_models[best_model_name]

    axes[1, 1].set_title(
        f"Top Drivers — {best_model_name}",
        fontweight="bold"
    )

    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(
            best_model.feature_importances_,
            index=X_test.columns
        ).sort_values(ascending=False).head(10)

    elif hasattr(best_model, "coef_"):
        importance = pd.Series(
            abs(best_model.coef_[0]),
            index=X_test.columns
        ).sort_values(ascending=False).head(10)

    else:
        axes[1, 1].text(
            0.5, 0.5,
            "Feature importance not available",
            ha="center", va="center"
        )
        importance = None

    if importance is not None:
        sns.barplot(
            x=importance.values,
            y=importance.index,
            ax=axes[1, 1]
        )
        axes[1, 1].set_xlabel("Importance")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
