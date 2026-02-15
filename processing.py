import pandas as pd
import re
import nltk
from tqdm import tqdm
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from newspaper import Article
from symspellpy import SymSpell
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
import pickle

nltk.download('brown', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
    
def process(df):
    # https://today.yougov.com/politics/articles/49552-trust-in-media-2024-which-news-outlets-americans-trust
    american_trusted_sources = ['weather.com', 'bbc.com', 'pbs.org', 'wsj.com',
                                        'forbes.com', 'abcnews.go.com', 'apnews.com', 'cbsnews.com',
                                        'time.com', 'espn.com', 'c-span.org', 'nbcnews.com', 'nytimes.com',
                                        'washingtonpost.com', 'usatoday.com', 'npr.org', 'ft.com', 'economist.com',
                                        'businessinsider.com', 'newsweek.com', 'theguardian.com', 'theatlantic.com', 'bloomberg.com',
                                        'newyorker.com', 'latimes.com', 'politco.com', 'news.yahoo.com', 'cnbc.com',
                                        'nypost.com', 'cnn.com', 'thehill.com', 'propublica.com']
    
    with open('model/random_forest_model.pkl', 'rb') as file:
        loaded_data = pickle.load(file)

    with open('data/positive-words.txt', 'r', encoding='latin-1') as f: # Specify encoding for positive words
        positive = f.read().splitlines()

    with open('data/negative-words.txt', 'r', encoding='latin-1') as f: # Specify encoding for negative words
        negative = f.read().splitlines()

    def overall_enrich_function(data):
        def txt_preprocessing(txt):
            reserved_stop_words = set(stopwords.words('english'))

            extra_stop_words=['one','two','three','four','five','six' "seven","eight","nine",'ten','using','sample','fig','figure','image','using']

            all_stop_words=list(reserved_stop_words.union(extra_stop_words))

            # Initialize lemmatizer
            lemmatizer = WordNetLemmatizer()

            if txt is None:
                return ""

            txt = txt.lower()  # lowercase
            txt = re.sub(r"^\s*([a-zA-Z]+(\s*\(.*?\))?\s*-\s*)", "", txt)  # Remove prefixes like "LOCATION (Source) -"
            txt = re.sub(r"^[a-zA-Z\s,]+(\s\([a-zA-Z]+\))?\s*-\s*", "", txt)
            txt = re.sub(r"<.*?>", " ", txt)  # Remove HTML tags
            txt = re.sub(r"[^a-zA-Z]", " ", txt)  # Remove special characters and digits
            txt = nltk.word_tokenize(txt)  # Tokenize text
            txt = [word for word in txt if word not in all_stop_words]  # Remove stopwords
            txt = [word for word in txt if len(word) >= 3]  # Remove words less than three letters
            txt = [lemmatizer.lemmatize(word, pos='v') for word in txt]  # Lemmatize words

            return " ".join(txt)  # return to string

        def dynamic_weighted_mean_similarity(data):
            # Normalize the cosine similarity scores to make them sum to 1 (so they can be used as weights)
            similarity_scores = data.values

            # Normalize the scores
            weights = similarity_scores / np.sum(similarity_scores, axis=1, keepdims=True)

            # Multiply each similarity score by its corresponding normalized weight
            weighted_scores = similarity_scores * weights

            # Sum the weighted scores for each row to get the dynamic weighted mean similarity
            weighted_mean = weighted_scores.sum(axis=1)

            return weighted_mean

        def extract_keywords(data, text_column, topn = 10, max_features = 10000, ngram_range=(1,4)):
            """
            Extract keywords from a DataFrame's text column using CountVectorizer and TfidfTransformer.

            Parameters:
            - data: pandas Dataframe
            - text_column: str, name of the column containing text data
            - topn: int, number of top keywords to return
            - max_features: int, maximum number of features to consider
            - ngram_range: tuple, range of n-grams to consider

            Returns:
            - pandas DataFrame with an additional 'keywords' column containing a list of extracted keywords
            """
            tqdm.pandas()

            #Initialize CountVectorzier
            cnt_vct = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
            word_cnt_vct = cnt_vct.fit_transform(data[text_column])

            # Initialize TfidfTransformer
            tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
            tfidf.fit(word_cnt_vct)

            # Helper Functions
            def sort_(matrix):
                tuples = list(zip(matrix.col, matrix.data))
                return sorted(tuples, key=lambda x: (-x[1], x[0]))

            def top_N(feature_names, sorted_items, topn=10):
                top_items = sorted_items[:topn]
                results = {}
                used_words = set()

                for idx, score in top_items:
                    feature_name = feature_names[idx]
                    # Skip adding if words are already part of a longer n-gram
                    if not any(word in used_words for word in feature_name.split()):
                        results[feature_name] = round(score, 3)
                        used_words.update(feature_name.split())

                return results

            def extract_keywords_for_row(text):
                tf_idf_vector = tfidf.transform(cnt_vct.transform([text]))
                sorted_items = sort_(tf_idf_vector.tocoo())
                feature_names = cnt_vct.get_feature_names_out()
                return top_N(feature_names, sorted_items, topn)

            data[f'{text_column}_keywords_with_score'] = data[text_column].progress_apply(extract_keywords_for_row)

            return data

        def extract_keywords_and_scores(df, title_column='title', content_column='clean_text', topn=10, max_features=10000, ngram_range=(1, 4)):
            """
            Extract keywords for both the title and the content columns and separate into lists.

            Args:
            df (pd.DataFrame): The input DataFrame containing title and content columns.
            title_column (str): The column name for the title.
            content_column (str): The column name for the content.
            topn (int): Number of top keywords to extract.
            max_features (int): Maximum number of features to consider.
            ngram_range (tuple): Range of n-grams to consider.

            Returns:
            pd.DataFrame: Updated DataFrame with keyword lists for both title and content.
            """

            # Extract keywords for the content
            df = extract_keywords(df, content_column, topn=topn, max_features=max_features, ngram_range=ngram_range)

            # Create separate keyword lists
            df[f'{content_column}_keyword_list'] = df[f'{content_column}_keywords_with_score'].apply(lambda x: list(x.keys()) if isinstance(x, dict) else [])

            return df

        def scrape_news_for_dataframe(df, sources_list = None):
            def search_news(clean_title, year, sources= sources_list):
                """
                Search news using Oxylabs API with the given title and year.
                """
                # Construct query with optional site filters
                if sources:
                    site_filter = " OR ".join([f"site:{source}" for source in sources])
                    query = f'{clean_title} {year} ({site_filter})'
                else:
                    query = f'{clean_title} {year}'
                print('\n------------------------------------\nq: ', query)
                payload = {
                    'source': 'google_search',
                    'query': query,
                    'parse': True,
                    'context': [
                        {'key': 'tbm', 'value': 'nws'},  # Specifies "News" tab in Google Search
                    ],
                    'pages': 1,
                    'limit': 5  # Limit to 5 results
                }
                try:
                    # Make POST request to Oxylabs API
                    response = requests.request(
                        'POST',
                        'https://realtime.oxylabs.io/v1/queries',
                        auth=('Johnny_l5htJ', 'Passwordnya_123'),
                        json=payload,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return data['results'][0]['content']['results']['main']  # Extract main results
                    else:
                        print(f"API Error: {response.status_code}, {response.text}")
                        return []
                except Exception as e:
                    print(f"Error during API request: {e}")
                    return []

            def fetch_full_content(url):
                """
                Fetch the full content of the article using newspaper3k.
                """
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    return article.text if article.text else None
                except Exception as e:
                    print(f"Error fetching article from {url}: {e}")
                    return None

            def clean_articles(main_results):
                """
                Clean and fetch the results from Oxylabs API using newspaper3k for article parsing.
                """
                cleaned_articles = []
                for item in main_results:
                    url = item.get('url')
                    if url:
                        # Fetch full content using newspaper3k
                        full_content = fetch_full_content(url)

                        # Validate full content
                        if full_content and len(full_content) > 200 and not any(
                            invalid_phrase in full_content.lower()
                            for invalid_phrase in [
                                "the requested article has expired",
                                "use your facebook account",
                                "all rights reserved"
                            ]
                        ):
                            cleaned_articles.append({
                                'title': item.get('title'),
                                'snippet': item.get('snippet', ''),
                                'full_content': full_content,
                                'url': url
                            })

                return cleaned_articles

            # Convert 'date' column from string to datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert with error handling
                if df['date'].isna().any():
                    print("Some dates could not be parsed. These rows will use only the title for searching.")

            # Process the DataFrame with tqdm
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Scraping news"):
                clean_title = row['clean_title']
                year = str(row['date'].year) if pd.notna(row['date']) else ""  # Use empty string if date is invalid

                # Search news using Oxylabs API
                main_results = search_news(clean_title, year)
                cleaned_articles = clean_articles(main_results)

                # Store up to 3 cleaned news articles in the DataFrame
                for i in range(1, 4):
                    if i <= len(cleaned_articles):
                        article = cleaned_articles[i - 1]
                        df.at[index, f"scraped_news_{i}_title"] = article.get('title', '')
                        df.at[index, f"scraped_news_{i}_url"] = article.get('url', '')
                        df.at[index, f"scraped_news_{i}_content"] = article.get('full_content', '')
                else:
                    df.at[index, f"scraped_news_{i}_title"] = ''
                    df.at[index, f"scraped_news_{i}_url"] = ''
                    df.at[index, f"scraped_news_{i}_content"] = ''

            return df

        def process_scraped_content_and_extract_keywords(data, scraped_columns, topn=10, max_features=10000, ngram_range=(1, 4)):
            """
            Cleans and extracts keywords for multiple scraped content columns.

            Parameters:
            - data: pd.DataFrame containing scraped content columns.
            - scraped_columns: List of column names to process (e.g., ['scraped_news_1_content', 'scraped_news_2_content', ...]).
            - topn: Number of top keywords to extract per column.
            - max_features: Maximum features for the CountVectorizer.
            - ngram_range: Tuple specifying n-gram range for keyword extraction.

            Returns:
            - pd.DataFrame with new keyword columns (keywords_1, keywords_2, etc.).
            """

            tqdm.pandas()

            def extract_keywords_from_text(text, cnt_vct, tfidf, topn):
                """Extract keywords from a single text."""
                if not text:
                    return {}
                tf_idf_vector = tfidf.transform(cnt_vct.transform([text]))
                sorted_items = sorted(zip(tf_idf_vector.tocoo().col, tf_idf_vector.tocoo().data), key=lambda x: -x[1])
                feature_names = cnt_vct.get_feature_names_out()
                results = {}
                used_words = set()
                for idx, score in sorted_items[:topn]:
                    feature_name = feature_names[idx]
                    if not any(word in used_words for word in feature_name.split()):
                        results[feature_name] = round(score, 3)
                        used_words.update(feature_name.split())
                return list(results.keys())

            # Preprocess text in each column
            for col in scraped_columns:
                print(f"Preprocessing column: {col}")
                data[f"clean_{col}"] = data[col].progress_apply(txt_preprocessing)

            # Combine all cleaned text columns for vectorization
            combined_texts = data[[f"clean_{col}" for col in scraped_columns]].fillna("").agg(" ".join, axis=1)
            
             # ===== SAFETY FIX — prevent empty vocabulary crash =====
                # If ALL scraped text is empty or whitespace, skip keyword extraction
            if not any(text.strip() for text in combined_texts):
                for i in range(1, len(scraped_columns)+1):
                    data[f"keywords_{i}"] = [[] for _ in range(len(data))]
                return data
            # =======================================================


            # Initialize CountVectorizer and TfidfTransformer
            print("Fitting CountVectorizer and TfidfTransformer...")
            cnt_vct = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
            tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
            word_cnt_vct = cnt_vct.fit_transform(combined_texts)
            tfidf.fit(word_cnt_vct)

            # Extract keywords for each column
            for i, col in enumerate(scraped_columns, 1):
                print(f"Extracting keywords for column: {col}")
                data[f"keywords_{i}"] = data[f"clean_{col}"].progress_apply(lambda x: extract_keywords_from_text(x, cnt_vct, tfidf, topn))

            # Drop intermediate cleaned columns if desired
            data.drop(columns=[f"clean_{col}" for col in scraped_columns], inplace=True)

            return data

        def calculate_keyword_similarity(data, keyword_list_col, keyword_cols):
            """
            Calculate cosine similarity between a keyword list and multiple keyword columns.

            Parameters:
            - data: pd.DataFrame containing the keyword columns.
            - keyword_list_col: str, name of the column containing the main keyword list.
            - keyword_cols: list of str, names of the columns to compare against the keyword list.

            Returns:
            - pd.DataFrame with additional columns for similarity scores.
            """
            # Initialize TfidfVectorizer
            tfidf_vectorizer = TfidfVectorizer()

            # Convert all keyword lists to strings
            data[keyword_list_col] = data[keyword_list_col].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
            for col in keyword_cols:
                data[col] = data[col].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

            for i, col in enumerate(keyword_cols, 1):
                # Combine keyword_list and the current column into one list for vectorization
                combined_texts = data[keyword_list_col].tolist() + data[col].tolist()

                # Check if combined_texts contains any meaningful content
                if not any(text.strip() for text in combined_texts):  # Check for empty strings or strings with only whitespace
                    # If no meaningful content, set similarity to 0 and continue to next column
                    data[f'similarity_score{i}'] = 0
                    continue

                # Vectorize the combined texts
                tfidf_matrix = tfidf_vectorizer.fit_transform(combined_texts)

                # Split the matrix into keyword_list vectors and current keyword column vectors
                keyword_list_vectors = tfidf_matrix[:len(data)]  # First half corresponds to keyword_list
                keyword_col_vectors = tfidf_matrix[len(data):]  # Second half corresponds to the current column

                # Calculate cosine similarity
                similarities = cosine_similarity(keyword_list_vectors, keyword_col_vectors).diagonal()

                # Add similarity scores to the DataFrame
                similarity_col_name = f'similarity_score{i}'
                data[similarity_col_name] = similarities

            return data

        def fakebert(df):
                
            # Load the saved model
            model = torch.load('model/fake_bert_model.pkl', map_location=torch.device('cpu'))

            # Load the BERT tokenizer
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

            # Function to predict the label of a list of texts
            def predict_texts(texts):
                results = []
                for text in texts:
                    # Tokenize the text
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

                    # Move inputs to the device (GPU if available, otherwise CPU)
                    if hasattr(model, 'bert') and hasattr(model.bert, 'device'):
                        inputs = {k: v.to(model.bert.device) for k, v in inputs.items()}
                    else:
                        inputs = {k: v.to('cpu') for k, v in inputs.items()}

                    # Make prediction
                    with torch.no_grad():
                        outputs = model(**inputs)
                        predicted_label = torch.argmax(outputs, dim=1).item()

                    # Append the result as a tuple (text, label)
                    results.append(predicted_label)

                return results

            df['fake_bert_prediction'] = predict_texts(df['clean_text'])

            return df

        def process_and_scrape_news(data, txt_preprocessing, extract_keywords, extract_keywords_and_scores, sources):
            """
            Consolidates text preprocessing, feature extraction, keyword extraction, and news scraping into one pipeline.

            Parameters:
            - data: pd.DataFrame containing at least a 'translated' column.
            - api_key: API key for the news scraping service.
            - news_api_url: URL endpoint for the news scraping service.
            - txt_preprocessing: Function to preprocess text data.
            - extract_keywords: Function to extract keywords from text.
            - extract_keywords_and_scores: Function to extract keywords and their scores.

            Returns:
            - Processed DataFrame with scraped news integrated.
            """
            # Step 1: Preprocess text
            print("Preprocessing text...")
            tqdm.pandas()
            data['clean_text'] = data['text'].progress_apply(lambda x: txt_preprocessing(x))
            data['clean_title'] = data['title'].progress_apply(lambda x: txt_preprocessing(x))

            # Step 2: Word Count Vectorization
            print("Extracting word count features...")
            cnt_vct = CountVectorizer(max_features=10000, ngram_range=(1, 4))
            if data['clean_text'].isnull().all() or data['clean_text'].str.strip().eq('').all():
                raise ValueError("Input text is empty after preprocessing. Please enter valid text.")
            word_cnt_vct = cnt_vct.fit_transform(data['clean_text'])

            # Step 3: TF-IDF Transformation
            print("Computing TF-IDF features...")
            tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
            tfidf.fit(word_cnt_vct)

            # Step 4: Extract Keywords
            print("Extracting keywords...")
            data = extract_keywords_and_scores(data) # This function uses the output of extract_keywords

            # Step 5: Scrape News
            print("Scraping news...")
            data = scrape_news_for_dataframe(data, sources_list = sources)

            # Step 6: Process Scraped News
            print("Processing scraped news...")

            print("Processing scraped news...")

            # 🔹 Ensure scraped content columns exist
            for i in range(1, 4):
                col = f"scraped_news_{i}_content"
                if col not in data.columns:
                    data[col] = ""

            scraped_columns = [f"scraped_news_{i}_content" for i in range(1, 4)]

            data = process_scraped_content_and_extract_keywords(
                data,
                scraped_columns,
                topn=10,
                max_features=10000,
                ngram_range=(1, 4)
            )

            # 🔹 Ensure similarity score columns exist
            for i in range(1, 4):
                col = f"similarity_score{i}"
                if col not in data.columns:
                    data[col] = 0.0

            # 🔹 Step 7: Averaging
            data['dynamic_weighted_mean_similarity'] = dynamic_weighted_mean_similarity(
                data[['similarity_score1', 'similarity_score2', 'similarity_score3']]
            )

            print("Pipeline completed!")
            return data


        def run_pipeline(data):
            """
            Run the entire pipeline: preprocess text, extract keywords, scrape news, and calculate keyword similarity.

            Parameters:
            - data: pd.DataFrame containing the text data in a 'text' column.
            - api_key: API key for the news scraping service.
            - news_api_url: URL endpoint for the news scraping service.

            Returns:
            - pd.DataFrame with processed text, extracted keywords, scraped news, and similarity scores.
            """
            tqdm.pandas()
            print("Step 1: Credibility Function")
            data = process_and_scrape_news(data, txt_preprocessing, extract_keywords, extract_keywords_and_scores, sources=american_trusted_sources)

            print("Step 2: Text Styled Analysis")
            # Style Analysis Functions
            sym_spell = SymSpell()
            sym_spell.create_dictionary(brown.words())

            def spell_checker(text):
                candidates = sym_spell.word_segmentation(text).corrected_string.split()
                words = text.split()
                if not words:
                    return 0
                score = sum(1 if word in candidates else 0 for word in words)
                return score / len(words)

            def lexical_diversity_rate_func(text):
                words = text.split()
                return (len(set(words)) / len(words)) if words else 0

            def sentiment_score_rate(text):
                words = text.lower().split()
                score = sum(1 if word in positive else -1 if word in negative else 0 for word in words)
                return score / len(words)

            def style_analysis(df):
                df['lexical_diversity_rate'] = df['clean_text'].apply(lexical_diversity_rate_func)
                df['spell_score'] = df['clean_text'].apply(spell_checker)
                df['sentiment_score'] = df['clean_text'].apply(sentiment_score_rate)
                return df

            # Enrich with style analysis feature
            data = style_analysis(data)

            print("Step 3: FakeBERT")

            # Enrich with FakeBERT result
            data = fakebert(data)

            print("Pipeline completed!")
            
            return data

        final = run_pipeline(data)

        return final


    final_enriched_data = overall_enrich_function(df)
    
    return final_enriched_data

def fake_deberta(df):

    with open("model/deberta_fake_news.pkl", "rb") as f:
        package = pickle.load(f)

    tokenizer = package["tokenizer"]
    model = package["model"]

    def predict(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        return pred   # 0 = real, 1 = fake

    df["fake_deberta_prediction"] = df["clean_text"].apply(predict)

    return df
