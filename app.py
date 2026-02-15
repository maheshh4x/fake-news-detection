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


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# ---------------- STYLING ---------------- #
st.markdown("""
<style>
.big-title {font-size: 40px; font-weight: 800; text-align:center;}
.small-sub {color: gray; font-size: 16px; margin-top: -10px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ---------------- TRUSTED DOMAINS ---------------- #
TRUSTED_DOMAINS = [
    "bbc.com", "reuters.com", "apnews.com", "thehindu.com",
    "ndtv.com", "timesofindia.indiatimes.com",
    "cnn.com", "nytimes.com", "washingtonpost.com"
]

GENERAL_FACT_KEYWORDS = [
    "who", "world health organization", "reuters", "ap news", "associated press",
    "vaccination", "verified coverage", "major outlets", "cdc", "unicef"
]

def check_domain(url: str):
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if any(td in domain for td in TRUSTED_DOMAINS):
            return "trusted", domain
        return "unknown", domain
    except:
        return "unknown", "unknown"


# ✅ Detect general fact statements (short + trusted org keywords)
def is_general_fact(text: str):
    t = text.lower().strip()
    return len(t.split()) < 40 and any(k in t for k in GENERAL_FACT_KEYWORDS)


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
    except:
        return text, "unknown"


def wiki_fact_check(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except:
        return None


# ---------------- HISTORY ---------------- #
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR SETTINGS ---------------- #
st.sidebar.title("⚙️ Extra Features")
mode = st.sidebar.radio("Mode", ["Relax ✅", "Strict 🔥"])
enable_translate = st.sidebar.checkbox("Auto Translate to English", value=True)
enable_wiki = st.sidebar.checkbox("Wikipedia Fact Check", value=False)

# ---------------- HEADER ---------------- #
st.markdown('<div class="big-title">📰 Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="small-sub">DeBERTa-v3 + Verification + Smart Tools</div>', unsafe_allow_html=True)
st.divider()

# ✅ CENTER ALIGN LAYOUT
col_center = st.columns([1, 6, 1])[1]

with col_center:
    st.subheader("🔍 Input News")

    tab1, tab2 = st.tabs(["✍️ Text", "🔗 URL"])

    news_text = ""
    domain_status = "unknown"
    domain_name = "N/A"
    url = ""

    with tab1:
        news_text = st.text_area(
            "Paste news content here:",
            height=220,
            placeholder="Paste full article for best accuracy..."
        )

    with tab2:
        url = st.text_input("Paste a News URL:")
        if url:
            domain_status, domain_name = check_domain(url)
            if domain_status == "trusted":
                st.success(f"✅ Trusted Domain: {domain_name}")
            else:
                st.warning(f"⚠️ Unknown Domain: {domain_name}")

            with st.spinner("Extracting content..."):
                try:
                    news_text = extract_text_from_url(url)
                    st.success("✅ Article extracted successfully!")
                    st.text_area("Preview:", news_text[:1200], height=150)
                except Exception as e:
                    st.error(f"Failed to extract: {e}")

    btn1, btn2 = st.columns(2)
    with btn1:
        verify_btn = st.button("✅ Verify", use_container_width=True)
    with btn2:
        clear_btn = st.button("🧹 Clear", use_container_width=True)

    if clear_btn:
        st.experimental_rerun()

# ---------------- PREDICT ---------------- #
if verify_btn:
    with col_center:
        if not news_text.strip():
            st.warning("⚠️ Please enter valid news content.")
        else:
            # ✅ Translate
            if enable_translate:
                translated_text, detected_lang = translate_to_english(news_text)
                if detected_lang not in ["en", "unknown"]:
                    st.info(f"🌍 Language detected: **{detected_lang}** → Translated to English ✅")
                news_text = translated_text

            cleaned_text = clean_text(news_text)

            # ✅ Model Prediction
            with st.spinner("🧠 Model predicting..."):
                result, confidence, real_prob, fake_prob = predict_news(cleaned_text)

            adjusted_label, adjusted_conf = strict_relax_decision(real_prob, fake_prob, mode)

            # ✅ Related articles FIX (BEST SEARCH QUERY)
            with st.spinner("🔎 Fetching related articles..."):
                query = " ".join(news_text.split()[:18])  # ✅ meaningful query
                related = fetch_related_articles(query)

            # ✅ Final verdict
            verdict = final_verdict(result, confidence, len(related), domain_status, news_text)


            # ✅ Save history
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

            # ✅ Results UI
            st.divider()
            st.subheader("📌 Result Dashboard")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🧠 Model", result)
            m2.metric("🧩 Mode Result", adjusted_label)
            m3.metric("✅ Verdict", verdict)
            m4.metric("📊 Confidence", f"{confidence*100:.2f}%")

            st.divider()
            st.subheader("🔥 Probability Breakdown")

            p1, p2 = st.columns(2)
            with p1:
                st.markdown("### ✅ Real Probability")
                st.progress(float(real_prob))
                st.write(f"**{real_prob*100:.2f}%**")

            with p2:
                st.markdown("### ❌ Fake Probability")
                st.progress(float(fake_prob))
                st.write(f"**{fake_prob*100:.2f}%**")

            st.subheader("📊 Probability Graph")
            chart_df = pd.DataFrame({"Probability": [real_prob, fake_prob]}, index=["Real", "Fake"])
            st.bar_chart(chart_df)

            # ✅ Summary
            st.divider()
            st.subheader("📝 Summary (Auto)")
            summary = simple_summary(news_text, max_sentences=3)
            st.info(summary)

            # ✅ Clickbait
            st.subheader("🚨 Clickbait Detector")
            cb_score, cb_level, cb_words = clickbait_score(news_text)
            st.write(f"**Clickbait Level:** {cb_level}")
            st.write(f"**Clickbait Score:** {cb_score}/100")
            if cb_words:
                st.write("**Detected Words:**", ", ".join(cb_words))
            else:
                st.write("✅ No strong clickbait words detected")

            # ✅ Credibility Score
            st.subheader("✅ Credibility Score (0–100)")
            cred_score = compute_credibility_score(
                real_prob=real_prob,
                fake_prob=fake_prob,
                related_count=len(related),
                domain_status=domain_status,
                clickbait_score=cb_score
            )
            st.metric("Credibility Score", f"{cred_score}")

            # ✅ Explain Why
            st.subheader("💡 Explain Why")
            reasons = explain_prediction(real_prob, fake_prob, len(related), domain_status, cb_level)
            for r in reasons:
                st.write("✅", r)

            # ✅ Sources
            st.divider()
            st.subheader("🔗 Verification Sources")
            if related:
                st.success(f"✅ Found {len(related)} related sources")
                for art in related:
                    st.write(f"- [{art['title']}]({art['link']})")
            else:
                st.warning("⚠️ No related sources found")

            # ✅ Wikipedia
            if enable_wiki:
                st.divider()
                st.subheader("📚 Wikipedia Fact Check")

                wiki_query = " ".join(cleaned_text.split()[:8])
                wiki_summary = wiki_fact_check(wiki_query)

                if wiki_summary:
                    st.success("✅ Wikipedia context found:")
                    st.write(wiki_summary)
                else:
                    st.info("No Wikipedia result found.")

            # ✅ TXT Report
            st.divider()
            st.subheader("📥 Download Report (TXT)")
            report_text = f"""
Fake News Detection Report
-------------------------
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Input Type: {"URL" if url else "Text"}
Domain: {domain_name}

Model Result: {result}
Mode Decision ({mode}): {adjusted_label}

Confidence: {confidence*100:.2f}%

Real Probability: {real_prob*100:.2f}%
Fake Probability: {fake_prob*100:.2f}%

Credibility Score: {cred_score}/100
Clickbait Level: {cb_level}
Related Sources Found: {len(related)}

Final Verdict: {verdict}

Summary:
{summary}
"""
            st.download_button("📄 Download TXT", data=report_text, file_name="news_report.txt")

            # ✅ Feedback
            st.divider()
            st.subheader("🧑‍💻 User Feedback")

            f1, f2 = st.columns(2)

            with f1:
                if st.button("✅ This is Correct"):
                    save_feedback("feedback.csv", news_text, verdict, "Correct")
                    st.success("Thanks! Feedback saved ✅")

            with f2:
                if st.button("❌ This is Wrong"):
                    save_feedback("feedback.csv", news_text, verdict, "Wrong")
                    st.error("Feedback saved ❌")

# ---------------- HISTORY TAB ---------------- #
with col_center:
    st.divider()
    st.subheader("🕘 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No history yet. Run prediction to see logs.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        st.download_button(
            "⬇️ Download History (CSV)",
            data=hist_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
