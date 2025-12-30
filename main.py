import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora, models
import matplotlib.pyplot as plt 
import seaborn as sns
#Packages required for this matter

nltk.download('punkt')
nltk.download('stopwords')

#OUR DATASETS AI NEWS HEADLINES
csv1 = pd.read_csv("csv_files\dataset_A_news_full_10500.csv")
csv2 = pd.read_csv("csv_files\dataset_B_news_subset_3500.csv")

#dataframe defined joint tabular datasets
df = pd.concat([csv1, csv2], ignore_index=True)

#Our work
stop_words = set(stopwords.words('english'))

#function to decaptialize text and tokenize
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

df['tokens'] = df['headline'].apply(preprocess)

all_tokens = [token for tokens in df['tokens'] for token in tokens]
word_freq = Counter(all_tokens)
print(word_freq.most_common(20))



#analyzing headline sentiment 
analyzer = SentimentIntensityAnalyzer()
#function for sentiment scores 
def get_sentiment_scores(text):
    scores = analyzer.polarity_scores(text)
    return scores['pos'], scores['neg'], scores['neu'], scores['compound'] #scores settled respected by positive, negative, and neutral

df[['pos','neg', 'neu', 'compound']] = df['headline'].apply(lambda x: pd.Series(get_sentiment_scores(x)))

dictionary = corpora.Dictinoary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]
lda_model = models.LdaMode(corpus, num_topics=5, id2word=dictionary, passes=10)
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")


df['fear_mongering'] = ((df['neg'] >= 0.5) & (df['compound'] <= -0.3))

df.to_csv("analyzed_headlines.csv", index=False)