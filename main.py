import numpy as np

from src.visualization import (
    save_forecast_plot,
    save_feature_importance,
    save_architecture_diagram
)


import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.data_loader import load_and_clean_data
from src.engineer import build_features

# Configuration
import os
import matplotlib.pyplot as plt

def save_architecture_diagram():
    plt.figure(figsize=(10,6))

    plt.text(0.5, 0.9, "AI Energy Forecasting System", ha='center', fontsize=14, weight='bold')

    plt.text(0.5, 0.75, "Raw Energy Dataset", ha='center')
    plt.text(0.5, 0.65, "↓ Data Cleaning", ha='center')
    plt.text(0.5, 0.55, "Feature Engineering", ha='center')
    plt.text(0.5, 0.45, "XGBoost Model Training", ha='center')
    plt.text(0.5, 0.35, "Forecast Prediction", ha='center')
    plt.text(0.5, 0.25, "Evaluation + RMSE/R²", ha='center')
    plt.text(0.5, 0.15, "Visualization Output", ha='center')

    plt.axis('off')
    plt.savefig("outputs/arch_diag.png")
    plt.close()

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "PJME_hourly.csv")
TARGET = 'PJME_MW'
SPLIT_DATE = '2015-01-01'
def run_pipeline():

    raw_df = load_and_clean_data(DATA_PATH)
    processed_df = build_features(raw_df, TARGET)

    train = processed_df.loc[processed_df.index < SPLIT_DATE]
    test = processed_df.loc[processed_df.index >= SPLIT_DATE]

    X_cols = [col for col in processed_df.columns if col != TARGET]

    X_train, y_train = train[X_cols], train[TARGET]
    X_test, y_test = test[X_cols], test[TARGET]

    #  MODEL CREATED HERE
    reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        objective='reg:squarederror'
    )

    reg.fit(X_train, y_train)

    # predictions
    test = test.copy()
    test["Prediction"] = reg.predict(X_test)

    # evaluation
    rmse = np.sqrt(mean_squared_error(y_test, test["Prediction"]))
    r2 = r2_score(y_test, test["Prediction"])

    print("RMSE:", rmse)
    print("R2:", r2)

    # CALL FUNCTIONS HERE (IMPORTANT)
    save_architecture_diagram()
    save_feature_importance(reg, X_train)
    save_forecast_plot(test)


# ENTRY POINT
if __name__ == "__main__":
    run_pipeline()