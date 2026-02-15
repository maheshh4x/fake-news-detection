import pandas as pd
import re
import streamlit as st
import numpy as np
import pickle
from assets import processing

# Load Model
with open('model/random_forest_model.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# ---------------- UI ----------------
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: green;
        color: white;
    }
    .true-class { color: green; }
    .fake-class { color: red; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fake News Detection")
st.write("_Disclaimer: The results are based on automated predictions and may not always be accurate._")

user_input_title = st.text_area("**Input news title:**", height=100)
user_input_text = st.text_area("**Input news text:**", height=200)

# Stop empty input
if (not user_input_title or not user_input_title.strip()) and (not user_input_text or not user_input_text.strip()):
    st.error("Please enter a news title or news text before continuing.")
    st.stop()

# Dataframe
df = pd.DataFrame({'title': [user_input_title], 'text': [user_input_text], 'date': [" "] })

# -------------- MAIN BUTTON --------------
if st.button("Continue"):

    with st.spinner("Processing..."):

        # Safety check
        cleaned_preview = re.sub(r'[^a-zA-Z ]', '', user_input_text).strip()
        if len(cleaned_preview.replace(" ", "")) < 3:
            st.error("News text is too short or invalid.")
            st.stop()

        # Columns expected by model
        columns_to_select = [
            'dynamic_weighted_mean_similarity',
            'spell_score',
            'lexical_diversity_rate',
            'sentiment_score',
            'fake_bert_prediction'
        ]

        # Run pipeline
        final_enriched_data = processing.process(df)

        # ----------- PREDICTION FIX -----------
        predictions = loaded_data.predict(final_enriched_data[columns_to_select])

        pred = predictions[0]   # extract label

        if pred == 1 or pred == 'Fake':
            string = "Fake"
        else:
            string = "True"

        st.write("Raw Model Output =", pred)
        # --------------------------------------

        # -------- RESULTS ----------
        st.header("Approach Results:")
        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)

        col1.metric("FakeBERT Prediction", final_enriched_data['fake_bert_prediction'].iloc[0])
        col2.metric("Lexical Diversity Rate", f"{final_enriched_data['lexical_diversity_rate'].iloc[0]:.2f}")
        col3.metric("Spell Score", f"{final_enriched_data['spell_score'].iloc[0]:.2f}")
        col4.metric("Sentiment Score", f"{final_enriched_data['sentiment_score'].iloc[0]:.2f}")
        col5.metric("Weighted Cosine Similarity", f"{final_enriched_data['dynamic_weighted_mean_similarity'].iloc[0]:.2f}")

        st.header("Final Result:")
        st.metric("Prediction", string)

        # -------- Ensure URL columns exist --------
        for i in range(1, 4):
            col = f"scraped_news_{i}_url"
            if col not in final_enriched_data.columns:
                final_enriched_data[col] = ""

        url_list = [
            final_enriched_data['scraped_news_1_url'].iloc[0],
            final_enriched_data['scraped_news_2_url'].iloc[0],
            final_enriched_data['scraped_news_3_url'].iloc[0]
        ]

        st.header("Related News:")

        for i, url in enumerate(url_list, 1):
            if url and str(url).strip() != "":
                st.markdown(f"[{url}]({url})")
                st.metric(
                    f"Cosine Similarity {i}",
                    f"{final_enriched_data.get(f'similarity_score{i}', pd.Series([0])).iloc[0]:.2f}"
                )
            else:
                st.write("No related article found")
