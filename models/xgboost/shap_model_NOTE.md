# shap_model.pkl

- Contents: a fitted `shap.TreeExplainer` (saved with `joblib.dump`) for the winning model: **xgboost**.
- Features it expects, in this exact order (15): amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, account_avg_amount, many_small_inbound_count, large_outbound_amount, new_device_flag, geo_velocity_spike, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER.
- Load with `joblib.load('models/xgboost/shap_model.pkl')`, then call `explainer.shap_values(X)` with a
  DataFrame holding those columns in that order.
- The path says `xgboost` because the path is frozen; the explainer is for whichever model won evaluation.
- The `.pkl` file itself is gitignored (models/**/*.pkl); this note is tracked.
