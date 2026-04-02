#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# Required for loading the saved training pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    pass


# =========================================================
# Maintenance switches
# =========================================================
DEBUG_MODE = False
APP_VERSION = "v1.0.0-single-file-maintenance"


# =========================================================
# Custom class used inside the saved joblib pipeline
# =========================================================
class FixedStructurePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, binary_idx=None, continuous_idx=None):
        self.binary_idx = binary_idx
        self.continuous_idx = continuous_idx

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.binary_idx_ = list(self.binary_idx) if self.binary_idx is not None else []
        self.continuous_idx_ = list(self.continuous_idx) if self.continuous_idx is not None else []

        if np.isnan(X).any():
            raise ValueError("Missing values detected in input data.")

        self.mean_ = {}
        self.std_ = {}
        for idx in self.continuous_idx_:
            col = X[:, idx]
            mean = np.mean(col)
            std = np.std(col)
            std = std if std > 0 else 1.0
            self.mean_[idx] = mean
            self.std_[idx] = std
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()

        if np.isnan(X).any():
            raise ValueError("Missing values detected in input data.")

        for idx in self.continuous_idx_:
            X[:, idx] = (X[:, idx] - self.mean_[idx]) / self.std_[idx]

        return X


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"
MODEL_PATH = ARTIFACT_DIR / "Best_Model_Final.joblib"
BUNDLE_PATH = ARTIFACT_DIR / "webapp_bundle_en.json"


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Clinical Risk Calculator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CSS
# =========================================================
def inject_css():
    st.markdown(
        dedent("""
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(102, 153, 255, 0.10), transparent 22%),
                radial-gradient(circle at top right, rgba(244, 166, 58, 0.08), transparent 20%),
                linear-gradient(180deg, #F5F8FD 0%, #F7F9FC 100%);
        }

        .block-container {
            max-width: 1080px;
            padding-top: 1.15rem;
            padding-bottom: 2.6rem;
        }

        .hero-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #FBFCFF 100%);
            border-radius: 30px;
            padding: 28px 34px;
            box-shadow: 0 18px 44px rgba(27, 39, 69, 0.08);
            border: 1px solid rgba(227, 234, 244, 0.95);
            margin-bottom: 18px;
        }

        .hero-title {
            font-size: 2.02rem;
            font-weight: 830;
            color: #22324A;
            margin-bottom: 0.22rem;
            letter-spacing: -0.03em;
        }

        .hero-subtitle {
            font-size: 0.98rem;
            color: #697A95;
            line-height: 1.58;
            margin-bottom: 0;
        }

        .intro-card {
            background: #FFFFFF;
            border-radius: 28px;
            padding: 22px 26px;
            box-shadow: 0 14px 34px rgba(27, 39, 69, 0.06);
            border: 1px solid rgba(227, 234, 244, 0.95);
            margin-bottom: 16px;
        }

        .intro-title {
            font-size: 1.2rem;
            font-weight: 790;
            color: #22324A;
            margin-bottom: 0.35rem;
        }

        .intro-note {
            color: #75839A;
            font-size: 0.94rem;
            line-height: 1.62;
        }

        div[data-testid="stForm"] {
            background: #FFFFFF !important;
            border: 1px solid rgba(227, 234, 244, 0.95) !important;
            border-radius: 28px !important;
            box-shadow: 0 14px 34px rgba(27, 39, 69, 0.06) !important;
            padding: 24px 24px 18px 24px !important;
            margin-bottom: 12px !important;
        }

        label[data-testid="stWidgetLabel"] p {
            color: #304058 !important;
            font-weight: 700 !important;
            font-size: 0.98rem !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: #F8FAFD !important;
            border: 1px solid #E6ECF4 !important;
            border-radius: 16px !important;
            min-height: 3rem !important;
            box-shadow: none !important;
        }

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
            border: 1px solid #92A7C7 !important;
            box-shadow: 0 0 0 3px rgba(83, 123, 255, 0.10) !important;
        }

        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(180deg, #5EA0FF 0%, #2E7BEF 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 16px !important;
            font-weight: 790 !important;
            font-size: 1rem !important;
            min-height: 3.1rem !important;
            box-shadow: 0 10px 22px rgba(46, 123, 239, 0.22) !important;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: linear-gradient(180deg, #4D94FA 0%, #226FE3 100%) !important;
        }

        div[data-testid="stButton"] button {
            background: #FFFFFF !important;
            color: #355071 !important;
            border: 1px solid #D7E1EE !important;
            border-radius: 16px !important;
            font-weight: 770 !important;
            min-height: 3.1rem !important;
        }

        div[data-testid="stButton"] button:hover {
            background: #F8FBFF !important;
            border: 1px solid #BFD0E8 !important;
        }

        .result-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #FCFDFF 100%);
            border-radius: 30px;
            padding: 30px 30px 28px 30px;
            box-shadow: 0 18px 44px rgba(27, 39, 69, 0.09);
            border: 1px solid rgba(227, 234, 244, 0.95);
            margin-top: 16px;
        }

        .result-label {
            color: #70809A;
            font-size: 1rem;
            font-weight: 780;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }

        .risk-number {
            font-size: 5rem;
            line-height: 0.95;
            font-weight: 860;
            margin-bottom: 0.65rem;
            letter-spacing: -0.04em;
        }

        .risk-band-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 10px 18px;
            border-radius: 999px;
            font-weight: 780;
            font-size: 1rem;
            margin-top: 0.15rem;
            margin-bottom: 1.2rem;
        }

        .risk-bar-wrap {
            margin-top: 1.25rem;
            margin-bottom: 0.95rem;
        }

        .risk-bar {
            position: relative;
            width: 100%;
            height: 22px;
            border-radius: 999px;
            background: linear-gradient(
                90deg,
                #34B369 0%,
                #7CCB4D 22%,
                #E8C948 50%,
                #F0A03A 74%,
                #F05A67 100%
            );
            box-shadow:
                inset 0 1px 2px rgba(0,0,0,0.10),
                0 6px 14px rgba(42, 58, 88, 0.08);
        }

        .risk-indicator {
            position: absolute;
            top: 50%;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #FFFFFF;
            border: 5px solid #24364F;
            transform: translate(-50%, -50%);
            box-shadow: 0 8px 18px rgba(36, 54, 79, 0.18);
        }

        .risk-scale {
            display: flex;
            justify-content: space-between;
            margin-top: 0.7rem;
            color: #63738D;
            font-size: 0.94rem;
            font-weight: 760;
        }

        .risk-description {
            color: #42536E;
            font-size: 1.04rem;
            line-height: 1.72;
            margin-top: 1.15rem;
            margin-bottom: 1rem;
            max-width: 92%;
        }

        .disclaimer-box {
            background: #FFF8EE;
            border: 1px solid #EFCB93;
            border-radius: 18px;
            padding: 16px 18px;
            color: #6B5842;
            font-size: 0.94rem;
            line-height: 1.62;
            margin-top: 1.1rem;
        }

        .summary-card {
            background: #FFFFFF;
            border-radius: 28px;
            box-shadow: 0 14px 34px rgba(27, 39, 69, 0.06);
            border: 1px solid rgba(227, 234, 244, 0.95);
            padding: 24px 26px 22px 26px;
            margin-top: 16px;
        }

        .section-title {
            font-size: 1.22rem;
            font-weight: 790;
            color: #22324A;
            margin-bottom: 1rem;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px 14px;
        }

        .summary-item {
            background: linear-gradient(180deg, #FAFCFE 0%, #F7FAFD 100%);
            border: 1px solid #E8EEF5;
            border-radius: 16px;
            padding: 12px 14px;
        }

        .summary-key {
            color: #72829A;
            font-size: 0.8rem;
            font-weight: 760;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .summary-value {
            color: #22324A;
            font-size: 0.98rem;
            font-weight: 780;
        }

        .small-note {
            color: #7E8CA1;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .error-note {
            background: #FFF4F4;
            border: 1px solid #F6CACA;
            color: #A14444;
            border-radius: 16px;
            padding: 12px 14px;
            margin-top: 12px;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .footer-note {
            color: #76849A;
            font-size: 0.92rem;
            text-align: center;
            margin-top: 1rem;
        }
        </style>
        """).strip(),
        unsafe_allow_html=True,
    )


# =========================================================
# Loaders
# =========================================================
@st.cache_data
def load_bundle():
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Bundle file not found: {BUNDLE_PATH}")
    with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


# =========================================================
# State
# =========================================================
def init_state():
    if "form_version" not in st.session_state:
        st.session_state.form_version = 0
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_form_values" not in st.session_state:
        st.session_state.last_form_values = None
    if "form_error" not in st.session_state:
        st.session_state.form_error = None


def do_reset():
    st.session_state.form_version += 1
    st.session_state.last_result = None
    st.session_state.last_form_values = None
    st.session_state.form_error = None


# =========================================================
# Helpers
# =========================================================
def get_theme(bundle):
    return bundle.get("ui", {}).get("theme", {})


def get_display_name(bundle, feature_name):
    return bundle["features"]["display_name_map"].get(
        feature_name,
        feature_name.replace("_", " ")
    )


def get_display_value_for_summary(bundle, feature_name, raw_value):
    cat_schema = bundle["features"].get("categorical", {})
    if feature_name in cat_schema:
        for opt in cat_schema[feature_name]["options"]:
            if str(opt["raw_value"]) == str(raw_value):
                return opt["display_value"]
    return format_summary_value(raw_value)


def get_risk_band(bundle, probability):
    p = float(probability)
    bands = bundle.get("ui", {}).get("risk_bands", [])
    if not bands:
        return {
            "key": "unknown",
            "label": "Unknown",
            "lower": 0.0,
            "upper": 1.0,
            "color": "#64748b",
            "description": "Risk band configuration is unavailable."
        }

    for band in bands:
        lower = float(band["lower"])
        upper = float(band["upper"])
        key = band.get("key", "")
        if key == "high":
            if lower <= p <= upper:
                return band
        else:
            if lower <= p < upper:
                return band
    return bands[-1]


def format_probability_display(probability):
    p = float(probability) * 100.0
    if p >= 99.95:
        return ">99.9%"
    if p <= 0.05:
        return "<0.1%"
    return f"{p:.1f}%"


def normalize_indicator_position(probability):
    p = float(probability)
    p = max(0.0, min(1.0, p))
    return p * 100.0


def extract_positive_probability(model, X):
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not expose predict_proba().")

    proba_vec = np.asarray(model.predict_proba(X)[0], dtype=float)

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = list(range(len(proba_vec)))

    if 1 in classes:
        pos_idx = classes.index(1)
    elif "1" in classes:
        pos_idx = classes.index("1")
    else:
        pos_idx = len(proba_vec) - 1

    pos_prob = float(proba_vec[pos_idx])
    return pos_prob, classes, proba_vec.tolist()


def build_feature_vector_from_form(bundle, form_values):
    feature_order = bundle["features"]["feature_order"]
    cat_schema = bundle["features"].get("categorical", {})
    cont_schema = bundle["features"].get("continuous", {})

    row = []

    for feat in feature_order:
        if feat in cat_schema:
            selected_raw = form_values[feat]

            if selected_raw == "__EMPTY__":
                display_name = get_display_name(bundle, feat)
                raise ValueError(f"Please select a value for {display_name}.")

            mapping = {
                str(opt["raw_value"]): int(opt["encoded_value"])
                for opt in cat_schema[feat]["options"]
            }

            if str(selected_raw) not in mapping:
                raise ValueError(f"Unexpected category for {feat}: {selected_raw}")

            row.append(float(mapping[str(selected_raw)]))

        elif feat in cont_schema:
            raw_val = str(form_values[feat]).strip()
            cfg = cont_schema[feat]
            min_val = float(cfg.get("min", 0.0))
            max_val = float(cfg.get("max", 100.0))
            display_name = get_display_name(bundle, feat)

            if raw_val == "":
                raise ValueError(f"Please enter a value for {display_name}.")

            try:
                numeric_val = float(raw_val)
            except Exception:
                raise ValueError(f"{display_name} must be a valid number.")

            if numeric_val < min_val or numeric_val > max_val:
                raise ValueError(
                    f"{display_name} must be between {min_val:g} and {max_val:g}."
                )

            row.append(float(numeric_val))

        else:
            raise ValueError(f"Feature {feat} not found in schema.")

    return np.array([row], dtype=float)


def format_summary_value(value):
    s = str(value).strip()
    if s == "":
        return "Empty"
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return f"{f:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return s


# =========================================================
# Renderers
# =========================================================
def render_header(bundle):
    text_cfg = bundle["ui"]["text"]
    html = dedent(f"""
    <div class="hero-card">
        <div class="hero-title">{text_cfg["app_title"]}</div>
        <p class="hero-subtitle">{text_cfg["app_subtitle"]}</p>
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)


def render_intro(bundle):
    text_cfg = bundle["ui"]["text"]
    html = dedent(f"""
    <div class="intro-card">
        <div class="intro-title">{text_cfg["input_section_title"]}</div>
        <div class="intro-note">
            Leave all fields blank on entry. Fill the values and submit. Use Reset to clear the current session state.
        </div>
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)


def render_result_card(bundle, probability, band):
    theme = get_theme(bundle)
    prob_text = format_probability_display(probability)
    indicator_pos = normalize_indicator_position(probability)
    indicator_pos = max(2.0, min(98.0, indicator_pos))
    risk_color = band.get("color", theme.get("primary_color", "#F4A63A"))

    html = dedent(f"""
    <div class="result-card">
        <div class="result-label">{bundle["ui"]["text"]["result_section_title"]}</div>
        <div class="risk-number" style="color:{risk_color};">{prob_text}</div>
        <div class="risk-band-chip" style="background:{risk_color}18;color:{risk_color};border:1px solid {risk_color}55;">
            {band["label"]}
        </div>
        <div class="risk-bar-wrap">
            <div class="risk-bar">
                <div class="risk-indicator" style="left:{indicator_pos:.1f}%;"></div>
            </div>
            <div class="risk-scale">
                <span>Low</span>
                <span>Intermediate</span>
                <span>High</span>
            </div>
        </div>
        <div class="risk-description">{band["description"]}</div>
        <div class="disclaimer-box">{bundle["ui"]["text"]["disclaimer"]}</div>
    </div>
    """).strip()

    st.markdown(html, unsafe_allow_html=True)


def render_placeholder_result():
    st.markdown(
        dedent("""
        <div class="result-card">
            <div class="result-label">Ready</div>
            <div style="font-size:1.15rem;font-weight:760;color:#22324A;margin-bottom:0.75rem;">
                Enter the patient information and calculate the risk.
            </div>
            <div class="small-note">
                All fields are intentionally blank on entry. Fill in the required values and submit the form to generate the estimated risk.
            </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def render_input_summary(bundle, form_values):
    feature_order = bundle["features"]["feature_order"]
    rows = []

    for feat in feature_order:
        display_name = get_display_name(bundle, feat)
        display_value = get_display_value_for_summary(bundle, feat, form_values[feat])
        rows.append(
            f"""
            <div class="summary-item">
                <div class="summary-key">{display_name}</div>
                <div class="summary-value">{display_value}</div>
            </div>
            """
        )

    rows_html = "".join(rows)

    html = dedent(f"""
    <div class="summary-card">
        <div class="section-title">Input Summary</div>
        <div class="summary-grid">{rows_html}</div>
    </div>
    """).strip()

    st.markdown(html, unsafe_allow_html=True)


def render_technical_details(bundle, result_dict):
    threshold = bundle.get("model", {}).get("binary_decision_threshold", 0.5)
    with st.expander("Technical Details", expanded=False):
        st.write("**Model**:", bundle.get("model", {}).get("display_name", "Unknown"))
        st.write("**Positive-class probability**:", f'{result_dict["probability"]:.10f}')
        st.write("**Display percent**:", format_probability_display(result_dict["probability"]))
        st.write("**Model classes_**:", result_dict["classes"])
        st.write("**All class probabilities**:", result_dict["all_probabilities"])
        st.write("**Binary decision threshold**:", threshold)


def render_debug_panel(bundle, result_dict, form_values):
    with st.expander("Diagnostics", expanded=False):
        st.write("**App version**:", APP_VERSION)
        st.write("**Model artifact path**:", str(MODEL_PATH))
        st.write("**Bundle path**:", str(BUNDLE_PATH))
        st.write("**Feature order**:", bundle["features"]["feature_order"])
        st.write("**Form values**:", form_values)
        st.write("**Encoded feature vector**:", result_dict["encoded_vector"])
        st.write("**Model classes_**:", result_dict["classes"])
        st.write("**All class probabilities**:", result_dict["all_probabilities"])
        st.write("**Positive-class probability**:", f'{result_dict["probability"]:.10f}')
        st.write("**Display percent**:", format_probability_display(result_dict["probability"]))
        st.write("**Risk band thresholds**:", bundle.get("ui", {}).get("risk_bands", []))
        st.write("**Binary decision threshold**:", bundle.get("model", {}).get("binary_decision_threshold", 0.5))


# =========================================================
# Main
# =========================================================
def main():
    init_state()
    inject_css()

    try:
        bundle = load_bundle()
        model = load_model()
    except Exception as e:
        st.error(f"Failed to initialize the app: {e}")
        st.stop()

    render_header(bundle)
    render_intro(bundle)

    feature_order = bundle["features"]["feature_order"]
    categorical = bundle["features"].get("categorical", {})
    continuous = bundle["features"].get("continuous", {})
    text_cfg = bundle["ui"]["text"]
    form_version = st.session_state.form_version

    with st.form(f"risk_form_{form_version}", clear_on_submit=False):
        form_values = {}

        for feat in feature_order:
            display_name = get_display_name(bundle, feat)

            if feat in categorical:
                options = categorical[feat]["options"]
                raw_values = [opt["raw_value"] for opt in options]
                display_lookup = {
                    str(opt["raw_value"]): opt["display_value"]
                    for opt in options
                }

                placeholder = "__EMPTY__"
                select_options = [placeholder] + raw_values
                display_lookup_with_placeholder = {placeholder: "Please select"}
                display_lookup_with_placeholder.update(display_lookup)

                selected_raw = st.selectbox(
                    label=display_name,
                    options=select_options,
                    index=0,
                    format_func=lambda x: display_lookup_with_placeholder.get(str(x), str(x)),
                    key=f"select_{feat}_{form_version}",
                )
                form_values[feat] = selected_raw

            elif feat in continuous:
                cfg = continuous[feat]
                min_val = float(cfg.get("min", 0.0))
                max_val = float(cfg.get("max", 100.0))
                unit = cfg.get("unit", "")

                label = display_name if not unit else f"{display_name} ({unit})"
                val = st.text_input(
                    label=label,
                    value="",
                    placeholder=f"{min_val:g} to {max_val:g}",
                    key=f"text_{feat}_{form_version}",
                )
                form_values[feat] = val

            else:
                st.error(f"Feature not found in schema: {feat}")
                st.stop()

        submitted = st.form_submit_button(
            text_cfg["submit_button_text"],
            use_container_width=True
        )

    reset_clicked = st.button(
        text_cfg.get("reset_button_text", "Reset"),
        use_container_width=True,
        key=f"reset_btn_{form_version}"
    )

    if reset_clicked:
        do_reset()
        st.rerun()

    if submitted:
        try:
            X = build_feature_vector_from_form(bundle, form_values)
            prob, classes, all_probas = extract_positive_probability(model, X)
            band = get_risk_band(bundle, prob)

            st.session_state.last_result = {
                "probability": prob,
                "band": band,
                "classes": classes,
                "all_probabilities": all_probas,
                "encoded_vector": X.tolist()[0],
            }
            st.session_state.last_form_values = form_values.copy()
            st.session_state.form_error = None

        except Exception as e:
            st.session_state.last_result = None
            st.session_state.last_form_values = None
            st.session_state.form_error = str(e)

    if st.session_state.form_error:
        st.markdown(
            f'<div class="error-note">{st.session_state.form_error}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.last_result is not None:
        render_result_card(
            bundle,
            st.session_state.last_result["probability"],
            st.session_state.last_result["band"],
        )
        if st.session_state.last_form_values is not None:
            render_input_summary(bundle, st.session_state.last_form_values)
        render_technical_details(bundle, st.session_state.last_result)

        if DEBUG_MODE:
            render_debug_panel(
                bundle,
                st.session_state.last_result,
                st.session_state.last_form_values
            )
    else:
        render_placeholder_result()

    st.markdown(
        dedent(f"""
        <div class="footer-note">
            {bundle["ui"]["text"]["footer_note"]}<br>
            {APP_VERSION}
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()