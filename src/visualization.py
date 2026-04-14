import matplotlib.pyplot as plt
import pandas as pd

def save_forecast_plot(test):
    plt.figure(figsize=(15, 6))

    plt.plot(test.index, test['PJME_MW'], label='Actual', alpha=0.7)
    plt.plot(test.index, test['Prediction'], label='Predicted', alpha=0.7)

    plt.title("Energy Forecasting: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Energy (MW)")
    plt.legend()

    plt.savefig("outputs/forecast_plot.png")
    plt.close()


def save_feature_importance(model, X_train):

    importance = model.feature_importances_
    features = X_train.columns

    df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10,6))
    plt.barh(df["Feature"][:10], df["Importance"][:10])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (XGBoost)")

    plt.savefig("outputs/feature_importance.png")
    plt.close()


def save_architecture_diagram():
    plt.figure(figsize=(10,6))

    plt.text(0.5, 0.9, "AI Energy Forecasting System", ha='center', fontsize=14, weight='bold')
    plt.text(0.5, 0.75, "Raw Data → Cleaning → Features", ha='center')
    plt.text(0.5, 0.6, "→ XGBoost Model Training", ha='center')
    plt.text(0.5, 0.45, "→ Prediction + Evaluation", ha='center')
    plt.text(0.5, 0.3, "→ Visualization Outputs", ha='center')

    plt.axis('off')
    plt.savefig("outputs/arch_diag.png")
    plt.close()