import streamlit as st
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse
from langdetect import detect
from deep_translator import GoogleTranslator
import wikipedia

from services.predictor import predict_news
from services.url_extractor import extract_text_from_url
from services.news_verifier import fetch_related_articles
from utils.text_cleaner import clean_text
from services.summary_generator import simple_summary
from services.explainability import clickbait_score, explain_prediction
from services.credibility_score import compute_credibility_score
from services.feedback_logger import save_feedback


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
        trusted = ["bbc.com", "reuters.com", "apnews.com", "thehindu.com",
                   "ndtv.com", "timesofindia.indiatimes.com",
                   "cnn.com", "nytimes.com", "washingtonpost.com"]
        if any(td in domain for td in trusted):
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
# PAGE CONFIG + STYLING (with button text visibility fix)
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeGuard • News Verifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary:     #6366f1;
        --primary-dark:#4f46e5;
        --bg:          #f9fafb;
        --card:        #ffffff;
        --text:        #111827;
        --text-light:  #374151;
        --text-meta:   #4b5563;
        --border:      #d1d5db;
        --border-strong:#9ca3af;
    }

    * {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: var(--bg);
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 1.8rem !important;
        padding-bottom: 4rem !important;
    }

    .header {
        background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%);
        color: white;
        padding: 3.8rem 2rem 3.2rem;
        border-radius: 0 0 32px 32px;
        text-align: center;
        margin: -1.5rem -5rem 2.8rem -5rem;
        box-shadow: 0 12px 48px rgba(99, 102, 241, 0.30);
    }

    .header h1 {
        font-size: 3.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1.2px;
    }

    .header .subtitle {
        font-size: 1.45rem;
        font-weight: 400;
        opacity: 0.95;
        margin-top: 0.9rem;
    }

    .card {
        background: var(--card);
        border-radius: 20px;
        padding: 2.1rem;
        margin-bottom: 2.2rem;
        border: 1px solid var(--border);
        box-shadow: 0 6px 24px rgba(0,0,0,0.07);
        transition: all 0.28s ease;
    }

    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.12);
    }

    .section-title {
        font-size: 1.48rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 1.5rem;
        padding-left: 1.1rem;
        border-left: 5px solid var(--primary);
    }

    p, div, span, label, .stMarkdown {
        color: var(--text) !important;
    }

    .stMarkdown a {
        color: var(--primary-dark) !important;
    }

    small, .stCaption, .stException, .stWarning {
        color: var(--text-meta) !important;
    }

    /* FIX: Buttons always readable - dark bg + white text */
    .stButton > button {
        color: white !important;
        background-color: #4f46e5 !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1.08rem !important;
        border-radius: 12px !important;
        height: 52px !important;
        transition: all 0.22s ease;
    }

    .stButton > button:hover {
        background-color: #4338ca !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(99, 102, 241, 0.35) !important;
    }

    /* Fix metric, progress, caption text visibility */
    div[data-testid="stMetric"] *, 
    .stProgress + div, 
    .stCaption, 
    .stMarkdown {
        color: #111827 !important;
    }

    .stProgress {
        background: #e5e7eb !important;
        border-radius: 10px;
    }

    .verdict-pill {
        display: inline-block;
        padding: 0.85rem 1.9rem;
        border-radius: 999px;
        font-size: 1.22rem;
        font-weight: 700;
        color: white;
        letter-spacing: -0.2px;
    }

    .real    { background: #059669; }
    .fake    { background: #dc2626; }
    .uncertain { background: #d97706; }

    section[data-testid="stSidebar"] > div {
        background: white !important;
        border-right: 1px solid var(--border) !important;
    }

    .sidebar-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 1.9rem;
    }

    hr {
        margin: 3.2rem 0 2rem !important;
        border-color: var(--border-strong) !important;
    }
    </style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)

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
    st.caption("FakeGuard • DeBERTa-v3 + multi-source verification")


# ────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🛡️ FakeGuard</h1>
    <div class="subtitle">Advanced Fake News Detection & Cross-Verification</div>
</div>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────
# MAIN CONTENT
# ────────────────────────────────────────────────
col = st.columns([1, 10, 1])[1]

with col:
    st.markdown('<div class="section-title">Analyze News</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Paste Text", "From URL"])

    news_text = ""
    url = ""
    domain_status = "unknown"
    domain_name = "—"

    with tab1:
        news_text = st.text_area(
            label="",
            height=280,
            placeholder="Paste the full news article here… (longer text = better accuracy)",
            key="text_area"
        )

    with tab2:
        url = st.text_input("News URL", placeholder="https://...", key="url_input")

        if url.strip():
            domain_status, domain_name = check_domain(url)
            if domain_status == "trusted":
                st.success(f"Trusted source • {domain_name}")
            else:
                st.info(f"Domain: {domain_name}")

            if st.button("Extract content"):
                with st.spinner("Extracting article…"):
                    try:
                        news_text = extract_text_from_url(url)
                        st.success("Article extracted successfully")
                        with st.expander("Article preview", expanded=True):
                            st.markdown(news_text[:1800] + "…")
                    except Exception as e:
                        st.error(f"Extraction failed: {str(e)}")

    b1, b2 = st.columns([5, 2])
    with b1:
        verify = st.button("🔍 Verify Now", type="primary", use_container_width=True)
    with b2:
        if st.button("Clear", use_container_width=True):
            st.rerun()


# ────────────────────────────────────────────────
# PREDICTION & RESULTS
# ────────────────────────────────────────────────
if verify and news_text.strip():
    with col:
        with st.spinner("Analyzing article… Please wait"):
            if enable_translate:
                translated_text, detected_lang = translate_to_english(news_text)
                if detected_lang not in ["en", "unknown"]:
                    st.info(f"🌍 Language detected: **{detected_lang}** → Translated to English ✅")
                news_text = translated_text

            cleaned_text = clean_text(news_text)

            result, confidence, real_prob, fake_prob = predict_news(cleaned_text)

            adjusted_label, adjusted_conf = strict_relax_decision(real_prob, fake_prob, mode)

            query = " ".join(news_text.split()[:18])
            related = fetch_related_articles(query)

            verdict = final_verdict(result, confidence, len(related), domain_status, news_text)

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

        # ── RESULTS ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

        cols = st.columns([2, 2, 2, 3])
        cols[0].metric("Model", result)
        cols[1].metric("Mode", adjusted_label)
        cols[2].metric("Confidence", f"{confidence*100:.0f}%")
        with cols[3]:
            if "REAL" in verdict:
                st.markdown(f'<div class="verdict-pill real">{verdict}</div>', unsafe_allow_html=True)
            elif "FAKE" in verdict:
                st.markdown(f'<div class="verdict-pill fake">{verdict}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-pill uncertain">{verdict}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Probability Breakdown</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Real News**")
            st.progress(real_prob)
            st.caption(f"{real_prob*100:.0f}%")
        with c2:
            st.markdown("**Fake News**")
            st.progress(fake_prob)
            st.caption(f"{fake_prob*100:.0f}%")

        st.subheader("Probability Graph")
        chart_df = pd.DataFrame({"Probability": [real_prob, fake_prob]}, index=["Real", "Fake"])
        st.bar_chart(chart_df)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Summary & Key Signals</div>', unsafe_allow_html=True)

        summary_text = simple_summary(news_text, max_sentences=3)
        st.markdown("**Summary**  \n" + summary_text)

        cb_score, cb_level, cb_words = clickbait_score(news_text)
        cred_score = compute_credibility_score(real_prob, fake_prob, len(related), domain_status, cb_score)

        cols = st.columns(2)
        with cols[0]:
            st.caption("Clickbait level")
            st.write(f"**{cb_level}**  ({cb_score}/100)")
            st.progress(cb_score / 100)
        with cols[1]:
            st.caption("Credibility score")
            st.metric("", f"{cred_score}/100")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── SOURCES & WIKIPEDIA ──
        if related or enable_wiki:
            with st.expander("Verification Sources & Context", expanded=True):
                if related:
                    st.success(f"Found {len(related)} related articles")
                    for art in related[:5]:
                        st.markdown(f"• [{art['title']}]({art['link']})")
                else:
                    st.warning("No corroborating sources found")

                if enable_wiki:
                    st.markdown("**Wikipedia context**")
                    wiki_text = wiki_fact_check(" ".join(cleaned_text.split()[:8]))
                    st.write(wiki_text or "— no relevant entry found —")

        # ── FEEDBACK + DOWNLOAD ──
        st.markdown('<div class="card">', unsafe_allow_html=True)

        fb1, fb2 = st.columns(2)
        with fb1:
            if st.button("✅ This result seems correct", use_container_width=True):
                save_feedback("feedback.csv", news_text, verdict, "Correct")
                st.success("Thank you!")
        with fb2:
            if st.button("❌ Something looks wrong", use_container_width=True):
                save_feedback("feedback.csv", news_text, verdict, "Wrong")
                st.success("Thank you for the feedback")

        # ── IMPORTANT: Define report_text here ──
        report_text = f"""FakeGuard Report ─ {datetime.now():%Y-%m-%d %H:%M}
═══════════════════════════════════════
Input:      {"URL" if url else "Text"}
Domain:     {domain_name}
Verdict:    {verdict}
Model:      {result}  ({confidence*100:.0f}%)
Mode:       {adjusted_label}
Real / Fake: {real_prob*100:.0f}% / {fake_prob*100:.0f}%
Credibility: {cred_score}/100
Clickbait:  {cb_level} ({cb_score}/100)
Sources:    {len(related)}

Summary:
{summary_text}
"""

        st.download_button(
            label="Download full report (.txt)",
            data=report_text,
            file_name=f"fakeguard_{datetime.now():%Y%m%d_%H%M}.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────
# HISTORY
# ────────────────────────────────────────────────
with col:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recent Checks</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No analyses yet. Try verifying an article!")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            df,
            column_config={
                "time": st.column_config.TextColumn("Time"),
                "input": st.column_config.TextColumn("Type"),
                "domain": "Domain",
                "model": "Model",
                "mode": "Mode",
                "confidence": st.column_config.NumberColumn("Conf", format="%.0f%%"),
                "real": st.column_config.NumberColumn("Real", format="%.0f%%"),
                "fake": st.column_config.NumberColumn("Fake", format="%.0f%%"),
                "verdict": "Verdict"
            },
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Export history (CSV)",
            df.to_csv(index=False).encode('utf-8'),
            "fakeguard_history.csv",
            "text/csv"
        )