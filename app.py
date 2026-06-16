import streamlit as st
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse
from langdetect import detect
from deep_translator import GoogleTranslator
import wikipedia

from config import TRUSTED_SOURCES
from services.predictor import predict_news
from services.url_extractor import extract_text_from_url
from services.news_verifier import fetch_related_articles
from utils.text_cleaner import clean_text
from services.summary_generator import simple_summary
from services.explainability import clickbait_score, explain_prediction
from services.credibility_score import compute_credibility_score
from services.feedback_logger import save_feedback
from services.pdf_report import generate_pdf_report


# ────────────────────────────────────────────────
# Initialize session state history
# ────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────

def check_domain(url: str):
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if any(td in domain for td in TRUSTED_SOURCES):
            return "trusted", domain
        return "unknown", domain
    except:
        return "unknown", "unknown"


def final_verdict(model_result, confidence, related_count, domain_status, original_text):
    HIGH = 0.80
    MID = 0.60
    if len(original_text.split()) < 25 and "recommend" in original_text.lower():
        return "REAL ✅ (General Fact Statement)"
    if related_count >= 2:
        return "REAL ✅ (Sources Verified)"
    if domain_status == "trusted" and model_result == "Real News" and confidence >= MID:
        return "REAL ✅ (Trusted Domain)"
    if model_result == "Fake News" and confidence >= HIGH and related_count == 0:
        return "FAKE ❌ (No Sources Found)"
    return "UNCERTAIN ⚠️ (Needs Manual Check)"


def strict_relax_decision(real_prob, fake_prob, mode):
    diff = abs(real_prob - fake_prob)
    if mode == "Strict 🔥" and diff < 0.15:
        return "Uncertain", max(real_prob, fake_prob)
    if fake_prob > real_prob:
        return "Fake News", fake_prob
    return "Real News", real_prob


def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            return translated, lang
        return text, "en"
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}")
        return text, "unknown"


def wiki_fact_check(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return None


# ────────────────────────────────────────────────
# PAGE CONFIG + STYLING (Premium Cyber-Security Theme)
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeGuard • News Verifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern custom fonts and aggressive dark theme styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #6366f1;
        --primary-glow: rgba(99, 102, 241, 0.15);
        --bg-main: #020617;
        --bg-card: rgba(15, 23, 42, 0.6);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border-color: rgba(255, 255, 255, 0.08);
    }

    * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Base App Overrides */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%) !important;
        color: var(--text-primary) !important;
    }

    /* Hide default streamlit header */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background: #090d16 !important;
        border-right: 1px solid var(--border-color) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        padding-top: 2rem !important;
    }
    .sidebar-title {
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }

    /* Glassmorphism Cards */
    .card {
        background: var(--bg-card) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .card:hover {
        border-color: rgba(99, 102, 241, 0.25) !important;
        box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.1) !important;
        transform: translateY(-2px) !important;
    }

    /* Titles */
    .section-title {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-bottom: 20px !important;
        padding-left: 12px !important;
        border-left: 4px solid var(--primary) !important;
        letter-spacing: -0.5px !important;
    }

    /* Custom Hero Banner */
    .hero-header-glass {
        background: rgba(30, 41, 59, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 20px !important;
        padding: 35px 20px !important;
        text-align: center !important;
        margin-bottom: 30px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2) !important;
    }
    .hero-title {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #3b82f6 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin: 0 !important;
        letter-spacing: -1.2px !important;
        text-shadow: 0 0 40px rgba(99, 102, 241, 0.2) !important;
    }
    .hero-subtitle {
        font-size: 1.05rem !important;
        color: var(--text-secondary) !important;
        margin-top: 8px !important;
        margin-bottom: 15px !important;
        font-weight: 400 !important;
    }

    /* Status Indicator */
    .status-indicator {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        background: rgba(16, 185, 129, 0.08) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        padding: 6px 14px !important;
        border-radius: 999px !important;
    }
    .pulse-green {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }

    /* Text and Elements Overrides */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] span {
        color: var(--text-secondary) !important;
        font-size: 0.95rem;
    }
    div[data-testid="stMarkdownContainer"] strong {
        color: #ffffff !important;
    }

    /* Form inputs & areas styling */
    div[class*="stTextArea"] textarea {
        background-color: rgba(15, 23, 42, 0.65) !important;
        color: #f8fafc !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        transition: all 0.25s ease !important;
        font-size: 0.95rem !important;
    }
    div[class*="stTextArea"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
    div[class*="stTextInput"] input {
        background-color: rgba(15, 23, 42, 0.65) !important;
        color: #f8fafc !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        transition: all 0.25s ease !important;
        font-size: 0.95rem !important;
        height: 46px !important;
    }
    div[class*="stTextInput"] input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }

    /* Input labels */
    div[class*="stTextArea"] label p, 
    div[class*="stTextInput"] label p,
    div[class*="stRadio"] label p,
    div[class*="stCheckbox"] label p {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Radio buttons & checkboxes custom */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 12px !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--border-color) !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
    }

    /* Checkbox styling */
    div[data-testid="stCheckbox"] label {
        color: #f8fafc !important;
    }

    /* Buttons visual overhaul */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        height: 50px !important;
        width: 100% !important;
        font-size: 1rem !important;
    }
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(0) !important;
    }

    /* Secondary Buttons */
    div[data-testid="stButton"] button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #cbd5e1 !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: none !important;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
    }

    /* Tab headers capsule styling */
    div[data-testid="stTabBar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
        gap: 8px !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stTabBar"] button {
        background: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        padding: 8px 20px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stTabBar"] button[aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        color: #a5b4fc !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
    }

    /* Expander custom dashboard card styling */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.2) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
        overflow: hidden;
    }
    div[data-testid="stExpander"] details {
        border: none !important;
    }
    div[data-testid="stExpander"] summary {
        background: transparent !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
        padding: 14px 18px !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #a5b4fc !important;
    }

    /* Clickbait badge words styling */
    .word-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        text-transform: capitalize;
    }

    /* Status alerts styling overrides */
    div[data-testid="stAlert"] {
        background: rgba(30, 41, 59, 0.35) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }

    /* Separator line style */
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin-top: 30px !important;
        margin-bottom: 30px !important;
    }
    </style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────
# VISUAL HELPER WIDGETS
# ────────────────────────────────────────────────

def get_verdict_card_html(verdict: str, model_label: str, confidence: float, mode_label: str) -> str:
    glow_color = "rgba(245, 158, 11, 0.35)"
    border_color = "rgba(245, 158, 11, 0.4)"
    bg_gradient = "linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(217, 119, 6, 0.12) 100%)"
    text_color = "#f59e0b"
    icon_svg = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'

    if "REAL" in verdict:
        glow_color = "rgba(16, 185, 129, 0.35)"
        border_color = "rgba(16, 185, 129, 0.4)"
        bg_gradient = "linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.12) 100%)"
        text_color = "#10b981"
        icon_svg = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
    elif "FAKE" in verdict:
        glow_color = "rgba(239, 68, 68, 0.35)"
        border_color = "rgba(239, 68, 68, 0.4)"
        bg_gradient = "linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(220, 38, 38, 0.12) 100%)"
        text_color = "#ef4444"
        icon_svg = '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="9" x2="15" y2="15"></line><line x1="15" y1="9" x2="9" y2="15"></line></svg>'

    return (
        '<div style="background:' + bg_gradient + ';border:1px solid ' + border_color + ';box-shadow:0 8px 32px 0 ' + glow_color + ';border-radius:16px;padding:24px;margin-bottom:24px;display:flex;flex-direction:column;align-items:center;text-align:center;">'
        '<div style="background:rgba(255,255,255,0.03);border-radius:50%;padding:16px;margin-bottom:12px;color:' + text_color + ';display:inline-flex;">'
        + icon_svg +
        '</div>'
        '<div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:1.5px;color:#94a3b8;font-weight:600;margin-bottom:6px;">Verification Verdict</div>'
        '<div style="font-size:1.5rem;font-weight:800;color:' + text_color + ';margin-bottom:18px;">' + verdict + '</div>'
        '<div style="display:flex;gap:16px;justify-content:center;width:100%;flex-wrap:wrap;border-top:1px solid rgba(255,255,255,0.08);padding-top:16px;">'
        '<div style="flex:1;min-width:100px;">'
        '<div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Classifier</div>'
        '<div style="font-size:1rem;font-weight:700;color:#ffffff;">' + model_label + '</div>'
        '</div>'
        '<div style="flex:1;min-width:100px;border-left:1px solid rgba(255,255,255,0.08);border-right:1px solid rgba(255,255,255,0.08);">'
        '<div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Confidence</div>'
        '<div style="font-size:1rem;font-weight:700;color:#ffffff;">' + str(int(confidence*100)) + '%</div>'
        '</div>'
        '<div style="flex:1;min-width:100px;">'
        '<div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;">Verify Mode</div>'
        '<div style="font-size:1rem;font-weight:700;color:#ffffff;">' + mode_label + '</div>'
        '</div>'
        '</div>'
        '</div>'
    )


def get_gauges_html(cred_score: float, confidence_pct: float) -> str:
    cred_offset = 238.76 - (cred_score / 100) * 238.76
    conf_offset = 238.76 - (confidence_pct / 100) * 238.76

    if cred_score >= 75:
        cred_color = "#34d399"
    elif cred_score >= 50:
        cred_color = "#fbbf24"
    else:
        cred_color = "#f87171"

    if confidence_pct >= 75:
        conf_color = "#60a5fa"
    else:
        conf_color = "#818cf8"

    return (
        '<div style="display:flex;gap:20px;justify-content:space-around;flex-wrap:wrap;margin-bottom:24px;">'

        '<div style="background:rgba(30,41,59,0.25);border:1px solid rgba(255,255,255,0.05);border-radius:16px;'
        'padding:20px;flex:1;min-width:160px;display:flex;flex-direction:column;align-items:center;">'
        '<div style="position:relative;width:100px;height:100px;display:flex;align-items:center;justify-content:center;">'
        '<svg width="100" height="100" viewBox="0 0 100 100" style="transform:rotate(-90deg);">'
        '<circle cx="50" cy="50" r="38" stroke="rgba(255,255,255,0.06)" stroke-width="7" fill="transparent"/>'
        '<circle cx="50" cy="50" r="38" stroke="' + cred_color + '" stroke-width="7" fill="transparent" '
        'stroke-dasharray="238.76" stroke-dashoffset="' + str(cred_offset) + '" stroke-linecap="round"/>'
        '</svg>'
        '<div style="position:absolute;font-size:1.4rem;font-weight:800;color:#ffffff;">' + str(int(cred_score)) + '</div>'
        '</div>'
        '<div style="margin-top:12px;font-size:0.8rem;font-weight:700;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.5px;">Credibility Score</div>'
        '</div>'

        '<div style="background:rgba(30,41,59,0.25);border:1px solid rgba(255,255,255,0.05);border-radius:16px;'
        'padding:20px;flex:1;min-width:160px;display:flex;flex-direction:column;align-items:center;">'
        '<div style="position:relative;width:100px;height:100px;display:flex;align-items:center;justify-content:center;">'
        '<svg width="100" height="100" viewBox="0 0 100 100" style="transform:rotate(-90deg);">'
        '<circle cx="50" cy="50" r="38" stroke="rgba(255,255,255,0.06)" stroke-width="7" fill="transparent"/>'
        '<circle cx="50" cy="50" r="38" stroke="' + conf_color + '" stroke-width="7" fill="transparent" '
        'stroke-dasharray="238.76" stroke-dashoffset="' + str(conf_offset) + '" stroke-linecap="round"/>'
        '</svg>'
        '<div style="position:absolute;font-size:1.4rem;font-weight:800;color:#ffffff;">' + str(int(confidence_pct)) + '%</div>'
        '</div>'
        '<div style="margin-top:12px;font-size:0.8rem;font-weight:700;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.5px;">Verification Conf.</div>'
        '</div>'

        '</div>'
    )


def get_probabilities_html(real_prob: float, fake_prob: float) -> str:
    real_pct = real_prob * 100
    fake_pct = fake_prob * 100
    return (
        '<div style="display:flex;flex-direction:column;gap:20px;margin-bottom:10px;">'
        '<div>'
        '<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.9rem;font-weight:600;">'
        '<span style="color:#34d399;display:flex;align-items:center;gap:6px;">'
        '<span style="width:8px;height:8px;background:#34d399;border-radius:50%;display:inline-block;"></span>'
        'Real News Probability</span>'
        '<span style="color:#34d399;font-weight:700;">' + "{:.1f}".format(real_pct) + '%</span>'
        '</div>'
        '<div style="height:12px;background:rgba(255,255,255,0.04);border-radius:6px;overflow:hidden;border:1px solid rgba(255,255,255,0.04);">'
        '<div style="height:100%;width:' + str(real_pct) + '%;background:linear-gradient(90deg,#10b981 0%,#34d399 100%);border-radius:6px;box-shadow:0 0 10px rgba(52,211,153,0.3);"></div>'
        '</div>'
        '</div>'
        '<div>'
        '<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.9rem;font-weight:600;">'
        '<span style="color:#f87171;display:flex;align-items:center;gap:6px;">'
        '<span style="width:8px;height:8px;background:#f87171;border-radius:50%;display:inline-block;"></span>'
        'Fake News Probability</span>'
        '<span style="color:#f87171;font-weight:700;">' + "{:.1f}".format(fake_pct) + '%</span>'
        '</div>'
        '<div style="height:12px;background:rgba(255,255,255,0.04);border-radius:6px;overflow:hidden;border:1px solid rgba(255,255,255,0.04);">'
        '<div style="height:100%;width:' + str(fake_pct) + '%;background:linear-gradient(90deg,#ef4444 0%,#f87171 100%);border-radius:6px;box-shadow:0 0 10px rgba(248,113,113,0.3);"></div>'
        '</div>'
        '</div>'
        '</div>'
    )


def get_clickbait_html(cb_score: float, cb_level: str) -> str:
    if cb_score >= 70:
        cb_color = "#ef4444"
        cb_bg = "linear-gradient(90deg,#ef4444 0%,#f87171 100%)"
    elif cb_score >= 35:
        cb_color = "#f59e0b"
        cb_bg = "linear-gradient(90deg,#f59e0b 0%,#fbbf24 100%)"
    else:
        cb_color = "#10b981"
        cb_bg = "linear-gradient(90deg,#10b981 0%,#34d399 100%)"

    return (
        '<div>'
        '<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.85rem;font-weight:600;">'
        '<span style="color:#cbd5e1;">Clickbait Level (' + cb_level + ')</span>'
        '<span style="color:' + cb_color + ';font-weight:700;">' + str(int(cb_score)) + '/100</span>'
        '</div>'
        '<div style="height:8px;background:rgba(255,255,255,0.04);border-radius:4px;overflow:hidden;border:1px solid rgba(255,255,255,0.04);">'
        '<div style="height:100%;width:' + str(cb_score) + '%;background:' + cb_bg + ';border-radius:4px;"></div>'
        '</div>'
        '</div>'
    )


def get_history_table_html(history_list: list) -> str:
    if not history_list:
        return '<div style="text-align:center;color:#64748b;padding:20px;">No verification history available.</div>'

    th = '<th style="padding:12px 10px;font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;">'
    thead = (
        '<thead><tr style="background:rgba(255,255,255,0.02);border-bottom:1px solid rgba(255,255,255,0.08);">'
        + th + 'Time</th>' + th + 'Type</th>' + th + 'Domain</th>'
        + th + 'Model</th>' + th + 'Conf</th>' + th + 'Verdict</th>'
        + '</tr></thead>'
    )

    rows = ''
    for item in history_list:
        verdict = item.get('final_verdict', 'UNCERTAIN')
        if 'REAL' in verdict:
            pill = '<span style="background:rgba(16,185,129,0.12);color:#34d399;border:1px solid rgba(16,185,129,0.25);padding:4px 10px;border-radius:999px;font-size:0.75rem;font-weight:700;">REAL</span>'
        elif 'FAKE' in verdict:
            pill = '<span style="background:rgba(239,68,68,0.12);color:#f87171;border:1px solid rgba(239,68,68,0.25);padding:4px 10px;border-radius:999px;font-size:0.75rem;font-weight:700;">FAKE</span>'
        else:
            pill = '<span style="background:rgba(245,158,11,0.12);color:#fbbf24;border:1px solid rgba(245,158,11,0.25);padding:4px 10px;border-radius:999px;font-size:0.75rem;font-weight:700;">UNCERTAIN</span>'
        td1 = '<td style="padding:12px 10px;color:#94a3b8;font-size:0.8rem;">' + item.get("time","") + '</td>'
        td2 = '<td style="padding:12px 10px;font-weight:600;font-size:0.85rem;color:#cbd5e1;">' + item.get("input_type","") + '</td>'
        td3 = '<td style="padding:12px 10px;color:#94a3b8;font-size:0.85rem;max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + item.get("domain","") + '</td>'
        td4 = '<td style="padding:12px 10px;font-weight:600;font-size:0.85rem;color:#ffffff;">' + item.get("model_result","") + '</td>'
        td5 = '<td style="padding:12px 10px;color:#a5b4fc;font-weight:600;font-size:0.85rem;">' + str(int(item.get("confidence",0))) + '%</td>'
        td6 = '<td style="padding:12px 10px;">' + pill + '</td>'
        rows += '<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">' + td1 + td2 + td3 + td4 + td5 + td6 + '</tr>'

    return (
        '<div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.08);border-radius:12px;background:rgba(15,23,42,0.2);">'
        '<table style="width:100%;border-collapse:collapse;text-align:left;">'
        + thead
        + '<tbody>' + rows + '</tbody>'
        + '</table></div>'
    )


# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛡️ Security Settings</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Verification mode",
        options=["Relax ✅", "Strict 🔥"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.divider()

    enable_translate = st.checkbox("Auto translate to English", value=True)
    enable_wiki = st.checkbox("Wikipedia fact-check", value=False)

    st.divider()
    st.caption("FakeGuard AI • DeBERTa-v3 model with real-time multi-source credibility verification core.")


# ────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────
st.markdown(
    '<div class="hero-header-glass">'
    '<div style="display:flex;align-items:center;justify-content:center;gap:14px;">'
    '<svg width="46" height="46" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter:drop-shadow(0 0 10px rgba(99,102,241,0.4));">'
    '<path d="M12 22C12 22 20 18 20 12V5L12 2L4 5V12C4 18 12 22 12 22Z" fill="#4f46e5" stroke="#818cf8" stroke-width="2" stroke-linejoin="round"/>'
    '<path d="M9 11L11 13L15 9" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
    '<h1 class="hero-title">FakeGuard AI</h1>'
    '</div>'
    '<div class="hero-subtitle">Advanced Fake News Detection &amp; Cross-Verification Platform</div>'
    '<div class="status-indicator">'
    '<span class="pulse-green"></span>'
    '<span style="color:#34d399;font-size:0.85rem;font-weight:600;letter-spacing:0.3px;">System Operational</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# ────────────────────────────────────────────────
# MAIN CONTENT
# ────────────────────────────────────────────────
col = st.columns([1, 10, 1])[1]

with col:
    st.markdown('<div class="section-title">Analyze News Core</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📄 Paste News Content", "🔗 Verify From URL"])

    news_text = ""
    url = ""
    domain_status = "unknown"
    domain_name = "—"

    with tab1:
        news_text = st.text_area(
            label="News Content",
            height=260,
            placeholder="Paste the full news article text here to trigger deep transformer analysis…",
            key="text_area",
            label_visibility="collapsed"
        )

    with tab2:
        url = st.text_input("Source URL", placeholder="https://news-source.com/article-slug", key="url_input")

        if url.strip():
            domain_status, domain_name = check_domain(url)
            if domain_status == "trusted":
                st.success(f"Trusted Source Identified: {domain_name}")
            else:
                st.info(f"Domain Extracted: {domain_name}")

            if st.button("Extract content"):
                with st.spinner("Scraping and parsing article content…"):
                    try:
                        news_text = extract_text_from_url(url)
                        st.success("Article text extracted successfully!")
                        with st.expander("Article preview", expanded=True):
                            st.markdown(news_text[:1800] + "…")
                    except Exception as e:
                        st.error(f"Failed to extract text: {str(e)}")

    b1, b2 = st.columns([5, 2])
    with b1:
        verify = st.button("🛡️ Run Verification Audit", type="primary", use_container_width=True)
    with b2:
        if st.button("Reset Dashboard", use_container_width=True):
            st.rerun()


# ────────────────────────────────────────────────
# PREDICTION & RESULTS
# ────────────────────────────────────────────────
if verify and news_text.strip():
    with col:
        with st.spinner("Analyzing text patterns, language traits & verifying sources..."):
            if enable_translate:
                translated_text, detected_lang = translate_to_english(news_text)
                if detected_lang not in ["en", "unknown"]:
                    st.info(f"🌍 Foreign Language Detected: **{detected_lang}** → Translated to English for processing.")
                news_text = translated_text

            cleaned_text = clean_text(news_text)

            # Predict using models (DeBERTa-v3 Core)
            result, confidence, real_prob, fake_prob = predict_news(cleaned_text)

            adjusted_label, adjusted_conf = strict_relax_decision(real_prob, fake_prob, mode)

            query = " ".join(news_text.split()[:18])
            related = fetch_related_articles(query)

            verdict = final_verdict(result, confidence, len(related), domain_status, news_text)

            # Store in session state
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_type": "URL" if url else "Text",
                "domain": domain_name,
                "model_result": result,
                "mode_result": adjusted_label,
                "confidence": round(confidence * 100, 2),
                "real_prob": round(real_prob * 100, 2),
                "fake_prob": round(fake_prob * 100, 2),
                "final_verdict": verdict
            })

        # ── VERDICT CARD ──
        st.markdown(get_verdict_card_html(verdict, result, confidence, adjusted_label), unsafe_allow_html=True)

        # ── GAUGES ──
        cb_score, cb_level, cb_words = clickbait_score(news_text)
        cred_score = compute_credibility_score(real_prob, fake_prob, len(related), domain_status, cb_score)
        st.markdown(get_gauges_html(cred_score, confidence * 100), unsafe_allow_html=True)

        # ── PROBABILITY BREAKDOWN ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Probability Breakdown</div>', unsafe_allow_html=True)
        st.markdown(get_probabilities_html(real_prob, fake_prob), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── SUMMARY & SIGNALS ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Summary & Key Signals</div>', unsafe_allow_html=True)

        summary_text = simple_summary(news_text, max_sentences=3)
        st.markdown(f"**AI Core Summary:**  \n{summary_text}")
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(get_clickbait_html(cb_score, cb_level), unsafe_allow_html=True)
            if cb_words:
                st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
                for w in cb_words:
                    st.markdown(f"<span class='word-badge'>{w}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; margin-bottom: 8px;'>AI Verification Indicators</div>", unsafe_allow_html=True)
            reasons = explain_prediction(real_prob, fake_prob, len(related), domain_status, cb_level)
            for r in reasons:
                st.markdown(f"""
                <div style="display: flex; gap: 8px; font-size: 0.85rem; color: #94a3b8; margin-bottom: 6px; align-items: flex-start;">
                    <span style="color: #6366f1;">•</span>
                    <span>{r}</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── SOURCES & WIKIPEDIA ──
        if related or enable_wiki:
            with st.expander("Verification Sources & Context", expanded=True):
                if related:
                    st.markdown(f"<div style='color: #34d399; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;'><span style='width: 6px; height: 6px; background: #34d399; border-radius: 50%;'></span>Found {len(related)} related references</div>", unsafe_allow_html=True)
                    for art in related[:5]:
                        st.markdown(f"""
                        <div style="
                            background: rgba(255, 255, 255, 0.02); 
                            border: 1px solid rgba(255, 255, 255, 0.05); 
                            border-radius: 8px; 
                            padding: 10px 14px; 
                            margin-bottom: 8px;
                        ">
                            <a href="{art['link']}" target="_blank" style="color: #a5b4fc !important; text-decoration: none; font-weight: 600; font-size: 0.85rem;">🔗 {art['title']}</a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: #f87171; font-weight: 600; margin-bottom: 12px;'>⚠️ No corroborating sources found</div>", unsafe_allow_html=True)

                if enable_wiki:
                    st.markdown("<div style='font-weight: 700; color: #ffffff; margin-top: 16px; margin-bottom: 8px;'>Wikipedia Context Fact-Check</div>", unsafe_allow_html=True)
                    wiki_text = wiki_fact_check(" ".join(cleaned_text.split()[:8]))
                    if wiki_text:
                        st.markdown(f"<div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; font-size: 0.9rem; line-height: 1.5; color: #cbd5e1;'>{wiki_text}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color: #64748b; font-style: italic;'>No relevant Wikipedia entry found.</div>", unsafe_allow_html=True)

        # ── FEEDBACK & DOWNLOAD REPORT ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fb1, fb2 = st.columns(2)
        with fb1:
            if st.button("✅ Feedback: Verdict is Accurate", use_container_width=True):
                save_feedback("feedback.csv", news_text, verdict, "Correct")
                st.success("Feedback saved. Thank you!")
        with fb2:
            if st.button("❌ Feedback: Verdict has Errors", use_container_width=True):
                save_feedback("feedback.csv", news_text, verdict, "Wrong")
                st.success("Feedback saved. Thank you for helping us improve!")

        report_text = f"""FakeGuard Report ─ {datetime.now():%Y-%m-%d %H:%M}
═══════════════════════════════════════
Input Type:   {"URL" if url else "Text"}
Source Domain: {domain_name}
Verdict:       {verdict}
Model Label:   {result} ({confidence*100:.0f}%)
Mode Adjusted: {adjusted_label}
Probability:   Real: {real_prob*100:.0f}% | Fake: {fake_prob*100:.0f}%
Credibility:   {cred_score}/100
Clickbait:     {cb_level} ({cb_score}/100)
Sources Count: {len(related)}

Summary:
{summary_text}
"""

        # Generate PDF report bytes
        import os
        pdf_data = {
            "Verification Report": f"FakeGuard ─ {datetime.now():%Y-%m-%d %H:%M}",
            "Input Type": "URL" if url else "Text",
            "Source Domain": domain_name,
            "Verdict": verdict,
            "Model Label": f"{result} ({confidence*100:.0f}%)",
            "Mode Adjusted": adjusted_label,
            "Real News Probability": f"{real_prob*100:.1f}%",
            "Fake News Probability": f"{fake_prob*100:.1f}%",
            "Credibility Score": f"{cred_score}/100",
            "Clickbait Level": f"{cb_level} ({cb_score}/100)",
            "Sources Count": str(len(related)),
            "Summary": summary_text
        }

        temp_pdf_path = f"temp_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        generate_pdf_report(pdf_data, temp_pdf_path)
        try:
            with open(temp_pdf_path, "rb") as f:
                pdf_bytes = f.read()
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="Download Full Report (.txt)",
                data=report_text,
                file_name=f"fakeguard_{datetime.now():%Y%m%d_%H%M}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="Download Full Report (PDF)",
                data=pdf_bytes,
                file_name=f"fakeguard_{datetime.now():%Y%m%d_%H%M}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────
# HISTORY
# ────────────────────────────────────────────────
with col:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recent Verification Audits</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No audit logs yet. Analyze news content to log results.")
    else:
        st.markdown(get_history_table_html(st.session_state.history), unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.history)
        st.download_button(
            label="Export Audit History (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="fakeguard_history.csv",
            mime="text/csv",
            use_container_width=True
        )