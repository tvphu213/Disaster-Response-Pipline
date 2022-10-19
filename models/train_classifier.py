from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
import re
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
# download onetime only
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """
    to load data from database sqlite and extract message df, categories df
    input: 
        database_filepath from user's input
    output: 
        X - message dataframe 
        y - categories dataframe

    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('MessageDetail', engine)
    category_columns = df.columns[4:]
    X = df.message
    y = df[category_columns]
    return X, y


def tokenize(text):
    """
    input:
        text: messages
    output:
        cleaned_tokens (list of words): nomalized, tokenzied text 
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    to build model 
    output: 
        cv - classification model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_leaf': [2, 4, 8],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)
    return cv


def evaluate_model(model, X_test, y_test):
    """
    input:
        model
        X_test: message df of test set
        y_test: category df of test set
    output: print out 3 metric below for each category of test set 
        f1 score 
        precision
        recall 
    """
    y_pred = model.predict(X_test)
    for index, feature in enumerate(y_test):
        print(f'Feature: {feature}')
        print(classification_report(y_test[feature], y_pred[:, index]))
    print(f'\nAccuracy: {(y_pred == y_test.values).mean()}')
    print(f'\nBest Parameters: {model.best_params_}')


def save_model(model, model_filepath):
    """
    to save trained model
    input 
        model: classification model
        model_filepath: model file's path
    output:
        pickle file save inputed path
    """
    with open(model_filepath, 'wb') as file:
        joblib.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
