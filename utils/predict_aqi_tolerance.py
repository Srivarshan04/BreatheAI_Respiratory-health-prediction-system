# utils/predict_aqi_tolerance.py
import os
import pickle
import logging
from typing import List, Optional, Tuple, Union, Iterable

import pandas as pd
import numpy as np

# optional imputer
try:
    from sklearn.impute import SimpleImputer
except Exception:
    SimpleImputer = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_FILENAME = os.path.join(_THIS_DIR, "logistic_regression_model.pkl")
DEFAULT_PREPROCESSING_FILENAME = os.path.join(_THIS_DIR, "preprocessing_columns.pkl")


def _load_artifacts(model_path: str = DEFAULT_MODEL_FILENAME, preprocessing_path: str = DEFAULT_PREPROCESSING_FILENAME):
    loaded_model = None
    loaded_preprocessing_columns = None

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)
            logger.info("Model loaded from %s", model_path)
        except Exception as e:
            logger.exception("Failed to load model from %s: %s", model_path, e)
    else:
        logger.warning("Model file '%s' not found.", model_path)

    if os.path.exists(preprocessing_path):
        try:
            with open(preprocessing_path, "rb") as f:
                loaded_preprocessing_columns = pickle.load(f)
            logger.info("Preprocessing columns loaded from %s", preprocessing_path)
        except Exception as e:
            logger.exception("Failed to load preprocessing columns from %s: %s", preprocessing_path, e)
    else:
        logger.warning("Preprocessing columns file '%s' not found.", preprocessing_path)

    return loaded_model, loaded_preprocessing_columns


def _normalize_severity(v: Optional[str]) -> str:
    if v is None:
        return "High"
    s = str(v).strip().lower()
    if s in ("mild", "low", "l"):
        return "Low"
    if s in ("moderate", "medium", "m"):
        return "Medium"
    if s in ("severe", "high", "h"):
        return "High"
    # fallback
    return "High"


def _to_title(x: str) -> str:
    try:
        return str(x).strip().title()
    except Exception:
        return str(x)


def predict_aqi_tolerance(
    disease: str,
    cough_severity: str,
    wheezing_severity: str,
    breathlessness_severity: str,
    chest_pain_severity: str,
    throat_irritation_severity: str,
    fatigue_severity: str,
    aqi: float,
    model=None,
    preprocessing_columns: Optional[List[str]] = None,
    *,
    model_path: str = DEFAULT_MODEL_FILENAME,
    preprocessing_path: str = DEFAULT_PREPROCESSING_FILENAME,
    return_proba: bool = False,
    impute_strategy: str = "none",
    fill_value: float = 0.0,
    tolerant_classes: Optional[Iterable[int]] = None
) -> Union[int, Tuple[int, Optional[float]]]:
    """
    Predict whether patient can tolerate given AQI.
    - tolerant_classes: iterable of class labels (ints) considered "can tolerate".
      Default: [1]. Set this to match how you trained your model.
    Returns:
      - if return_proba==False: int 0/1 (0=Cannot tolerate, 1=Can tolerate)
      - if return_proba==True: (int, float_probability_of_tolerant_decision_or_None)
    Notes:
      - This function attempts to infer model.feature_names_in_ first. If unavailable, it uses preprocessing_columns.
      - It maps incoming severities ('mild','moderate','severe') -> 'Low'|'Medium'|'High' and builds expected one-hot style columns.
      - The returned probability (when available) is the probability mass that corresponds to `tolerant_classes`.
    """
    if model is None or preprocessing_columns is None:
        loaded_model, loaded_preproc = _load_artifacts(model_path=model_path, preprocessing_path=preprocessing_path)
        if model is None:
            model = loaded_model
        if preprocessing_columns is None:
            preprocessing_columns = loaded_preproc

    if model is None:
        raise ValueError("Model not available for prediction.")
    # determine expected columns
    expected_cols = None
    try:
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(getattr(model, "feature_names_in_"))
            logger.info("Using model.feature_names_in_ (count=%d)", len(expected_cols))
        else:
            # common positions in pipeline
            if hasattr(model, "named_steps"):
                for name, step in model.named_steps.items():
                    if hasattr(step, "feature_names_in_"):
                        expected_cols = list(getattr(step, "feature_names_in_"))
                        logger.info("Using feature_names_in_ from pipeline step '%s' (count=%d)", name, len(expected_cols))
                        break
            if expected_cols is None and hasattr(model, "steps"):
                for _, step in model.steps:
                    if hasattr(step, "feature_names_in_"):
                        expected_cols = list(getattr(step, "feature_names_in_"))
                        logger.info("Using feature_names_in_ from pipeline steps (count=%d)", len(expected_cols))
                        break
    except Exception:
        logger.exception("Could not introspect feature_names_in_, fallback to preprocessing_columns")

    if expected_cols is None:
        if preprocessing_columns is not None:
            expected_cols = list(preprocessing_columns)
            logger.info("Using provided preprocessing_columns (count=%d)", len(expected_cols))
        else:
            raise RuntimeError("Unable to determine expected feature columns. Provide preprocessing_columns or a model with feature_names_in_.")

    # normalize severities
    cough = _normalize_severity(cough_severity)
    wheeze = _normalize_severity(wheezing_severity)
    breath = _normalize_severity(breathlessness_severity)
    chest = _normalize_severity(chest_pain_severity)
    throat = _normalize_severity(throat_irritation_severity)
    fat = _normalize_severity(fatigue_severity)
    disease_title = _to_title(disease) if disease else ""

    # build mapping-friendly dict of inputs
    inputs = {
        "cough_severity": cough,
        "wheezing_severity": wheeze,
        "breathlessness_severity": breath,
        "chest_pain_severity": chest,
        "throat_irritation_severity": throat,
        "fatigue_severity": fat,
        "disease": disease_title,
        "aqi": aqi
    }

    # Build row with defaults then populate
    row = {}
    for col in expected_cols:
        # handle aqi (case-insensitive)
        if col.lower() == "aqi":
            try:
                row[col] = float(aqi) if aqi is not None and str(aqi).strip() != "" else np.nan
            except Exception:
                row[col] = np.nan
            continue

        # one-hot severity pattern e.g. cough_severity_category_Low
        # support both patterns: <symptom>_severity_category_<Value> OR <symptom>_severity_<Value>
        lowered = col.lower()
        if "_severity_category_" in lowered or lowered.endswith("_severity_low") or lowered.endswith("_severity_medium") or lowered.endswith("_severity_high"):
            # try to extract base symptom and target value
            if "_severity_category_" in col:
                # crude but workable split
                parts = col.split("_")
                target_val = parts[-1]
                base_name = "_".join(parts[:-2])
            else:
                parts = col.split("_")
                target_val = parts[-1]
                base_name = "_".join(parts[:-2])

            # try to match base_name to known inputs keys
            matched = None
            for key in ("cough_severity", "wheezing_severity", "breathlessness_severity", "chest_pain_severity", "throat_irritation_severity", "fatigue_severity"):
                if key in base_name or base_name in key or base_name.replace("_", "") in key.replace("_", ""):
                    matched = key
                    break
            if matched:
                try:
                    row[col] = 1.0 if str(inputs[matched]).lower() == str(target_val).lower() else 0.0
                except Exception:
                    row[col] = 0.0
            else:
                # default
                row[col] = 0.0
            continue

        # disease one-hot like condition_Asthma
        if lowered.startswith("condition_") or lowered.startswith("disease_"):
            rhs = col.split("_", 1)[1]
            try:
                row[col] = 1.0 if disease_title and disease_title == rhs else 0.0
            except Exception:
                row[col] = 0.0
            continue

        # otherwise try direct mapping of known keys (case-insensitive)
        if col in inputs:
            row[col] = inputs[col]
            continue
        if col.lower() in inputs:
            row[col] = inputs[col.lower()]
            continue
        # fallback safe numeric default
        row[col] = 0.0

    input_df = pd.DataFrame([row], columns=expected_cols)

    logger.debug("Built input_df columns: %s", list(input_df.columns))
    # show columns that were filled with default (0.0) which may indicate mapping issue
    defaults = [c for c in input_df.columns if input_df.loc[0, c] in (0.0, 0, None) and c.lower() != "aqi"]
    if defaults:
        logger.debug("Columns using default fallback (potential unmapped inputs): %s", defaults)

    # coerce numeric types where possible
    for c in input_df.columns:
        try:
            input_df[c] = pd.to_numeric(input_df[c])
        except Exception:
            pass

    # handle NaNs / imputation as before
    nan_cols = input_df.columns[input_df.isna().any()].tolist()
    if nan_cols:
        logger.warning("Prediction input contains NaNs: %s", nan_cols)
        strat = (impute_strategy or "none").lower()
        if strat == "none":
            raise ValueError(f"Missing/invalid values in input for columns: {nan_cols}")
        if SimpleImputer is None:
            raise RuntimeError("Imputation requested but sklearn SimpleImputer not available.")
        if strat in ("median", "mean"):
            imputer = SimpleImputer(strategy=strat)
            input_df[:] = imputer.fit_transform(input_df)
        elif strat == "constant":
            imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
            input_df[:] = imputer.fit_transform(input_df)
        else:
            raise ValueError(f"Unknown impute_strategy: {impute_strategy}")

    if input_df.isna().any().any():
        remaining = input_df.columns[input_df.isna().any()].tolist()
        raise ValueError(f"Input contains NaN for columns after imputation attempt: {remaining}")

    # Predict using model
    try:
        # prepare tolerant_classes default & helper sets
        if tolerant_classes is None:
            tolerant_classes = [1]
        # create tolerant match sets (ints and string-lower)
        tolerant_ints = set()
        tolerant_strs = set()
        for tc in tolerant_classes:
            try:
                tolerant_ints.add(int(tc))
            except Exception:
                tolerant_strs.add(str(tc).lower())

        # get probabilities if available
        proba = None
        proba_tolerant = None
        pred_label = None

        if hasattr(model, "predict_proba"):
            proba_all = model.predict_proba(input_df)
            probs = proba_all[0]  # ndarray of probabilities for each class (in order of model.classes_)
            classes = getattr(model, "classes_", None)

            # If model.classes_ exists, use it to determine which indices correspond to tolerant classes.
            # If not, treat indices (0..n-1) as class labels.
            indices_tolerant = []
            if classes is not None:
                # iterate and collect indices whose class value matches tolerant_classes
                for j, cls_val in enumerate(classes):
                    matched = False
                    try:
                        if int(cls_val) in tolerant_ints:
                            matched = True
                    except Exception:
                        pass
                    if not matched:
                        if str(cls_val).lower() in tolerant_strs:
                            matched = True
                    if matched:
                        indices_tolerant.append(j)
                # Compute total probability mass for tolerant indices
                try:
                    proba_tolerant = float(np.sum([probs[i] for i in indices_tolerant])) if indices_tolerant else 0.0
                except Exception:
                    proba_tolerant = None
                # predicted class: choose argmax and map via classes
                try:
                    idx = int(np.argmax(probs))
                    try:
                        pred_label = int(classes[idx])
                    except Exception:
                        # if cannot cast to int, set raw class value
                        pred_label = classes[idx]
                except Exception:
                    pred_label = None
            else:
                # No classes provided: treat argmax index as label (index)
                try:
                    idx = int(np.argmax(probs))
                    pred_label = int(idx)
                    # tolerant_classes likely refer to labels; if a tolerant class equals index, include it
                    proba_tolerant = float(probs[idx]) if idx in tolerant_ints else 0.0
                except Exception:
                    pred_label = None
                    proba_tolerant = None

            proba = proba_tolerant if proba_tolerant is not None else None
            logger.debug("predict_proba output (first row): %s", probs)
        else:
            # no predict_proba - fallback to predict
            pred_raw = model.predict(input_df)
            if hasattr(pred_raw, "__len__"):
                try:
                    pred_label = int(pred_raw[0])
                except Exception:
                    pred_label = pred_raw[0]
            else:
                try:
                    pred_label = int(pred_raw)
                except Exception:
                    pred_label = pred_raw
            proba = None
            proba_tolerant = None

        # Now convert pred_label (which may be multi-class) into binary decision (0/1)
        # Default: tolerant_classes == [1] (common for binary models)
        if tolerant_classes is None:
            tolerant_classes = [1]

        # Decision: 1 if predicted class maps to tolerant_classes, else 0
        decision = 0
        try:
            # Try integer comparison
            try:
                if pred_label is not None and int(pred_label) in tolerant_ints:
                    decision = 1
                else:
                    # try string match
                    if pred_label is not None and str(pred_label).lower() in tolerant_strs:
                        decision = 1
            except Exception:
                # fallback: string match
                if pred_label is not None and str(pred_label).lower() in tolerant_strs:
                    decision = 1
                else:
                    decision = 0
        except Exception:
            decision = 0

        # If caller asked for probability of "can tolerate", return proba (which we computed as proba_tolerant)
        if return_proba:
            # Ensure proba is a float in 0..1 or None
            try:
                if proba is not None:
                    proba_val = float(proba)
                    # clamp possible tiny numeric issues
                    if proba_val < 0.0:
                        proba_val = 0.0
                    if proba_val > 1.0:
                        # if model returned percentages (unlikely), convert if >1 by dividing by 100 if >1 and <=100
                        if proba_val <= 100.0:
                            proba_val = proba_val / 100.0
                        else:
                            # leave as-is but warn
                            logger.debug("Returned probability out of expected range: %s", proba_val)
                    # ---------- DEBUG LOG ADDED ----------
                    try:
                        logger.info("predict_aqi_tolerance -> pred_label=%s decision=%s proba=%s classes=%s", pred_label, decision, proba_val, getattr(model, "classes_", None))
                    except Exception:
                        pass
                    # -------------------------------------
                    return decision, proba_val
                else:
                    try:
                        logger.info("predict_aqi_tolerance -> pred_label=%s decision=%s proba=None classes=%s", pred_label, decision, getattr(model, "classes_", None))
                    except Exception:
                        pass
                    return decision, None
            except Exception:
                try:
                    logger.info("predict_aqi_tolerance -> pred_label=%s decision=%s proba_exception classes=%s", pred_label, decision, getattr(model, "classes_", None))
                except Exception:
                    pass
                return decision, None
        else:
            # ---------- DEBUG LOG ADDED ----------
            try:
                logger.info("predict_aqi_tolerance -> pred_label=%s decision=%s proba=%s classes=%s", pred_label, decision, proba, getattr(model, "classes_", None))
            except Exception:
                pass
            # -------------------------------------
            return decision

    except Exception as e:
        logger.exception("Prediction failure: %s", e)
        raise
