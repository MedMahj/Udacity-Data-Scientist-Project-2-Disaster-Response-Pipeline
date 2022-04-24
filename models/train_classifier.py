
# import libraries
import re
import sys
import time
import datetime
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# download necessary NLTK data
nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    """
       Function:
       load data from database
       Args:
       database_filepath: the path of the database
       Return:
       X (DataFrame) : Message features dataframe
       Y (DataFrame) : target dataframe
       category_names (list of str) : target labels list
       """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = df.columns[4:]

    return X,Y,category_names
    

def tokenize(text):
    """
    Function: split text into words and return the root form of the words
    Args:
      text(str): the message
    Return:
      clean_tokens(list of str): a list of the root form of the message words
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop = stopwords.words("english")
    tokens = [tok for tok in tokens if tok not in stop]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
     Function: build a model for classifing the disaster messages
     Return:
       cv: classification model
     """
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # Create Grid search parameters
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 60, 70]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: Evaluate the model and print the f1 score, precision and recall for each output category 
    Args:
     model: the classification model
     X_test: test messages
     Y_test: test target
    Return:
     report : evauluation report for each output category 
    """

    Y_pred = model.predict(X_test)

    report = {}
    for i in range(len(category_names)):
        report[category_names[i]] = classification_report(Y_test[:,i], Y_pred[:,i], zero_division= 1)
        print('{} :\n {}\n'.format(category_names[i].upper(),report[category_names[i]]))
    
    accuracy = (Y_pred == Y_test).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))
    
    return report


def save_model(model, model_filepath):
    """
    Function: Save a pickle file of the model
    Args:
     model: the classification model
     model_filepath (str): the path of pickle file
    """

    pickle.dump(model, open(model_filepath, 'wb'))
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        print(datetime.datetime.now())
        start = time.time()
        print('Training model...')
        model.fit(X_train, Y_train)
        end = time.time()
        print(datetime.datetime.now())
        print('Model Trained in ', str(datetime.timedelta(seconds=(end-start))))
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()