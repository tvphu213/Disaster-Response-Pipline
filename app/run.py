import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import nltk
# download onetime only
# nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

app = Flask(__name__)


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


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageDetail', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    category_list = list(df.columns[4:])
    category_count = df[category_list].sum(axis=0).reset_index()
    category_distribute = pd.DataFrame({'category_names':category_list, 'category_count':list(category_count.iloc[:,1])}).sort_values('category_count', ascending=False)
    
    category_list_with_genre = list(df.columns[3:])
    category_list_with_genre = df[category_list_with_genre].groupby('genre').agg({sum}).reset_index()
    direct_category_distribute = list(category_list_with_genre.iloc[0,1:].reset_index()[0])
    news_category_distribute = list(category_list_with_genre.iloc[1,1:].reset_index()[1])
    social_category_distribute = list(category_list_with_genre.iloc[2,1:].reset_index()[2])
    graphs = [
        {
            'data': [
                Bar( 
                    x=category_distribute.category_names,
                    y=category_distribute.category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=category_list,
                    y=direct_category_distribute,
                    name='direct'
                ),
                Bar(
                    x=category_list,
                    y=news_category_distribute,
                    name='news'
                ),
                 Bar(
                    x=category_list,
                    y=social_category_distribute,
                    name='social'
                )
            ],

            'layout': {
                'title': 'Categories by Genre',
                'barmode': 'stack',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }, 
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
