import pandas as pd
from langdetect import detect
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


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
    INPUT_PATH = 'data/ratings.csv'
    OUTPUT_PATH = 'data/ratings_clean.parquet'

    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f'Error: Could not find input file at {INPUT_PATH}')
        exit(1)

    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
    except Exception as e:
        print('Error: Unable to download the necessary documents for tokenization: {e}')

    required_columns = ['review', 'review_id', 'review_rating', 'review_length']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f'Error: Missing required columns: {", ".join(missing)}')
        exit(1)
    
    

    df.drop_duplicates(inplace=True, subset=['review'])
    df.drop_duplicates(inplace=True, subset=['review_id'])
    df.dropna(inplace=True, subset=['review', 'review_id'])

    
   
    print(f'Rows before filtering: {len(df)}')
    df=limit_review_length(df, 10, 500)
    print(f'Rows after filtering: {len(df)}')
    
    try:
        df = df[df['review'].apply(is_english)]
        print(f'Rows after english check: {len(df)}')
    except Exception as e:
        print(f'Error: There was an issue determining if one or many of the reviews are english: {e}')
        exit(1)
    
    try:
        df['review_clean'] = df['review'].apply(clean_text)
    except Exception as e:
        print(f'Error: There was an issue cleaning the review data: {e}')
        exit(1)


    df['tokens'] = df['review_clean'].str.split()

    stop_words = set(stopwords.words('english'))
    negations = {'no', 'not', 'never', 'neither', 'nor', 'none', "don't", "won't", "can't", "isn't", "aren't"}
    stop_words = stop_words - negations  
    
    if not stop_words:
        print('There are no stopwords, exiting')
        exit(1)

    df['tokens'] = df['tokens'].apply(lambda x: [w for w in x if w not in stop_words])

    lemmatizer= WordNetLemmatizer()
    df['tokens'] = df['tokens'].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])


    df['review_processed'] = df['tokens'].apply(lambda x: ' '.join(x))

    final_df = df[['review_id', 'review_rating', 'review_processed', 'review_length', 'review_clean']]
    final_df.to_parquet(OUTPUT_PATH, index=False)