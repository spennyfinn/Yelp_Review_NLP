import pandas as pd
from langdetect import detect
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pyarrow

def limit_review_length(df: pd.DataFrame, min_words:int, max_words:int):
    return df[(df['review_length']>=min_words) &(df['review_length']<=max_words)]

def is_english(text:str):
    try:
        return detect(text)=='en'
    except:
        return False

def clean_text(text):
    text=text.lower()
    text = re.sub(r'http\S+', '', text)
    text= re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text




if __name__ == '__main__':
    nltk.download('wordnet')
    nltk.download('stopwords')

    df = pd.read_csv('/Users/spennyfinn/Documents/Projects/NLP/data/ratings.csv')

    df.drop_duplicates(inplace=True, subset=['review'])
    df.drop_duplicates(inplace=True, subset=['review_id'])

    df.dropna(inplace=True, subset=['review', 'review_id'])

    

    print(f'Rows before filtering: {len(df)}')
    df=limit_review_length(df, 10, 500)
    print(f'Rows after filtering: {len(df)}')

    df = df[df['review'].apply(is_english)]
    print(f'Rows after english check: {len(df)}')

    df['review_clean'] = df['review'].apply(clean_text)

    df['tokens'] = df['review_clean'].str.split()

    stop_words = set(stopwords.words('english'))
    negations = {'no', 'not', 'never', 'neither', 'nor', 'none', "don't", "won't", "can't", "isn't", "aren't"}
    stop_words = stop_words - negations  

    df['tokens'] = df['tokens'].apply(lambda x: [w for w in x if w not in stop_words])


    lemmatizer= WordNetLemmatizer()
    df['tokens'] = df['tokens'].apply(lambda x: (lemmatizer.lemmatize(w) for w in x))


    df['review_processed'] = df['tokens'].apply(lambda x: ' '.join(x))

    final_df = df[['review_id', 'review_rating', 'review_processed', 'review_length', 'review_clean']]
    final_df.to_parquet('/Users/spennyfinn/Documents/Projects/NLP/data/ratings_clean.parquet', index=False)





    

