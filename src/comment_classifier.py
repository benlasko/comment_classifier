import pandas as pd
import numpy as np

import spacy
import re
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, precision_score, recall_score

spacy_lemmatizer = spacy.load('en_core_web_sm')

StopWords = set(stopwords.words('english'))


def dataframe_overview(df):
    '''
    Prints the following for a quick overview of a dataframe:
        Information returned from .info()
        First five rows (.head())
        Shape (.shape)
        List of column names
        Information returned from .describe(). The "top" is the most common value. The "freq" is the most common value’s frequency.
        Total duplicate rows
    Parameter
    ----------
    df:  pd.DataFrame
        A Pandas DataFrame
    Returns
    ----------
       None.
    '''
    columns = df.columns.values.tolist()

    print("\u0332".join("INFO "))
    print(f'{df.info()}\n\n')
    print("\u0332".join("HEAD "))
    print(f'{df.head()}\n\n')
    print("\u0332".join("SHAPE "))
    print(f'{df.shape}\n\n')
    print("\u0332".join("COLUMNS "))
    print(f'{columns}\n\n')
    print("\u0332".join("COLUMN STATS "))
    print(f'{df.describe()}\n\n')
    print('\u0332'.join("TOTAL DUPLICATE ROWS "))
    print(f' {df.duplicated().sum()}')


def fit_and_score_model(model, X_train, y_train, X_test, y_test, weighted=True):
    '''
    Fits train/test data to a model and prints accuracy, precision, recall, and f1 scores.
    Parameter
    ----------
    model:  model object
        A model to fit and score.
    X_train:  arr
        X_train data.
    y_train:  arr
        y_train data.
    X_test:  arr
        X_test data.
    y_test:  arr
        y_test data.
    weighted:  boolean
        If True then for precision, recall and f1 the average='weighted'.  Default = True.
    Returns
    ----------
    None.
    '''
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    if weighted==True:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')


def custom_stopwords(stop_words, additional_stopwords):
    '''
    Creates a new stopwords set with additional stopwords added to the original stopwords.
    Parameters
    ----------
    stop_words:  set
        Original set of stopwords to add new words to.
    additional_stopwords:  set or list
        New stopwords to add to the original stopwords.
    Returns
    ----------
    A new stopwords set with all original and additional stopwords.
    '''
    add_stopwords = set(additional_stopwords)
    StopWords = stop_words.union(add_stopwords)
    return set(StopWords)


def lowercase_text(text):
    '''
    Lowercases text.
    Parameter
    ----------
    text: str
        Text to lowercase.
    Returns
    ----------
    Lowercased text.
    '''
    return text.lower()


def remove_nums_and_punctuation(text):
    '''
    Removes numbers and puncuation from text.
    Parameter
    ----------
    text: str
        Text to remove numbers and puncuation from.
    Returns
    ----------
    Text with numbers and puncuation removed.
    '''
    punc = '!()-[]{};:\\,<>./?@#$%^&*_~;1234567890'
    for ch in text:
        if ch in punc:
            text = text.replace(ch, '')
    return text


def remove_newlines(text):
    '''
    Removes new lines from text.
    Parameter
    ----------
    text: str
        Text to remove new lines from.
    Returns
    ----------
    Text with new lines removed.
    '''
    text.replace('\n', '')
    return text


def remove_urls(text):
    '''
    Removes URLs from text.
    Parameter
    ----------
    text: str
        Text to remove URLs from.
    Returns
    ----------
    Text with URLs removed.
    '''
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text).split())


def lemmatize_string(string):
    '''
    Lemmatize all words in a string using the spaCy lemmatizer.
    Parameter
    ----------
    text:  str
        Text to lemmatize.
    Returns
    ----------
    List with each word replaced by its lemma.
    '''
    lem_string = spacy_lemmatizer(string)
    lemmatized = ' '.join([w.lemma_ for w in lem_string])
    return lemmatized


def string_to_word_list(text):
    '''
    Splits text into a list of words.
    Parameter
    ----------
    text: str
        Text to create list of words from.
    Returns
    ----------
    List of words from the text.
    '''
    return text.split(' ')


def remove_stopwords(word_lst, stop_words):
    '''
    Removes stopwords from text.
    Parameters
    ----------
    word_lst: list
        List of words from which to remove stopwords.
    stop_words: set or list
        Stopwords to remove from the list of words.
    Returns
    ----------
    List of words with stopwords removed.
    '''
    return [word for word in word_lst if word not in stop_words]


def word_list_to_string(word_lst):
    '''
    Creates a string with all words from a list of words.
    Parameter
    ----------
    word_lst: list
        List of words to join into a string.
    Returns
    ----------
    String of the words in the word list passed in.
    '''
    return ' '.join(word_lst)


def text_cleaning_pipeline(text, stop_words=StopWords):
    '''
    A text cleaning pipeline to clean a string by spellchecking, lowercasing, removing numbers/puncuation/new lines/urls/stopwords, and lemmatizing.
    Parameters
    ----------
    text:  str
        The text to be cleaned.
    stop_words:  set
        Stopwords to remove from the text.
    Returns
    ----------
    String of the cleaned text.
    '''
    text_tb = TextBlob(text)
    text_sp = text_tb.correct()
    text_lc = lowercase_text(text_sp)
    text_np = remove_nums_and_punctuation(text_lc)
    text_nnls = remove_newlines(text_np)
    text_nurl = remove_urls(str(text_nnls))
    text_lemd = lemmatize_string(text_nurl)
    words = string_to_word_list(text_lemd)
    words_nsw = remove_stopwords(words, stop_words)
    cleaned_str = word_list_to_string(words_nsw)
    return cleaned_str


if __name__ == '__main__':

    train_data = pd.read_csv('data/train.tsv', sep='\t', header=None)
    test_data = pd.read_csv('data/test.tsv', sep='\t', header=None)

    df = pd.concat([train_data, test_data], axis=0)
    df = df.drop(columns=[2])
    df.columns = ['comment', 'emotions']


    single_label = []

    for multi_label in df.emotions:
        single_label.append(multi_label.split(',')[0])

    single_label_series = pd.Series(single_label)

    df['emotion'] = single_label_series

    df = df.drop(columns=['emotions'])

    df.loc[df['emotion'].isin(['3', '11', '14', '2']), 'emotion'] = 'Anger'
    df.loc[df['emotion'].isin(['6', '19', '22', '26', '13']), 'emotion'] = 'Excitement'
    df.loc[df['emotion'].isin(['1', '7', '20', '21', '23', '17']), 'emotion'] = 'Joy'
    df.loc[df['emotion'].isin(['0', '4', '5', '8', '15', '18']), 'emotion'] = 'Love'
    df.loc[df['emotion'].isin(['27']), 'emotion'] = 'Neutral'
    df.loc[df['emotion'].isin(['9', '10', '12', '16', '24', '25']), 'emotion'] = 'Sadness'


    additional_stopwords = {'thing', 'think', 'would', 'get', 'th', 'nt', 'go', 'name', 'I', 'man', 'one', 'oh'}

    StopWords = custom_stopwords(StopWords, additional_stopwords)


    df['comment'] = df['comment'].apply(text_cleaning_pipeline)


    df = df.drop_duplicates(subset='comment')


    X = df['comment']
    y = df['emotion']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

    tfv = TfidfVectorizer()

    X_train = tfv.fit_transform(X_train).toarray()
    X_test = tfv.transform(X_test).toarray()

    xgb = XGBClassifier(n_estimators=400, learning_rate=.32, max_depth=5, n_jobs=-1)


    fit_and_score_model(xgb, X_train, y_train, X_test, y_test)


    def comment_classifier(string):
        '''
        Prints the top three choices for which emotion (Anger, Excitement, Joy, Love, Neutral, or Sadness) the string is classified as and how certain the model is for each of those choices.
        Parameter
        ----------
        string:  str
            Text to classify.
        Returns
        ----------
        None.
        '''
        clean_text = text_cleaning_pipeline(string, StopWords)
        text_list = [clean_text]
        transformed = tfv.transform(text_list)
        probas = xgb.predict_proba(transformed)
        inds = np.argsort(probas[0])[::-1]

        if inds[:1] == 0:
            label_1 = 'Anger'
        if inds[:1] == 1:
            label_1 = 'Excitement'
        if inds[:1] == 2:
            label_1 = 'Joy'
        if inds[:1] == 3:
            label_1 = 'Love'
        if inds[:1] == 4:
            label_1 = 'Neutral'
        if inds[:1] == 5:
            label_1 = 'Sadness'

        if inds[1:2] == 0:
            label_2 = 'Anger'
        if inds[1:2] == 1:
            label_2 = 'Excitement'
        if inds[1:2] == 2:
            label_2 = 'Joy'
        if inds[1:2] == 3:
            label_2 = 'Love'
        if inds[1:2] == 4:
            label_2 = 'Neutral'
        if inds[1:2] == 5:
            label_2 = 'Sadness'

        if inds[2:3] == 0:
            label_3 = 'Anger'
        if inds[2:3] == 1:
            label_3 = 'Excitement'
        if inds[2:3] == 2:
            label_3 = 'Joy'
        if inds[2:3] == 3:
            label_3 = 'Love'
        if inds[2:3] == 4:
            label_3 = 'Neutral'
        if inds[2:3] == 5:
            label_3 = 'Sadness'

        confidence_level_1 = probas[0][inds[:1]][0]
        confidence_level_2 = probas[0][inds[1:2]][0]
        confidence_level_3 = probas[0][inds[2:3]][0]

        print(f'Class: {label_1}.  Certainty: {confidence_level_1}%.\n')
        print(f'Class: {label_2}.  Certainty: {confidence_level_2}%.\n')
        print(f'Class: {label_3}.  Certainty: {confidence_level_3}%.')
