from newspaper import Article

def extract_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text
