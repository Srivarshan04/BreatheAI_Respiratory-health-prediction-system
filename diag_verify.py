# diag_verify.py
import importlib, app, pandas as pd
import os, sys
importlib.reload(app)
print("Using app.py at:", app.__file__)
try:
    # load dataset the same way app does
    df = None
    dp = os.path.join(os.path.dirname(app.__file__), "breatheai_aqi_10yr_dataset_enriched.csv")
    if os.path.exists(dp):
        df = pd.read_csv(dp)
        df.columns = df.columns.str.strip().str.lower()
    ok, msg = app.verify_forecast_artifacts(df, T_in=7)
    print("verify_forecast_artifacts -> ok:", ok, " msg:", msg)
except Exception as e:
    import traceback; traceback.print_exc()
    print("Exception while running verify:", e)
