# Udacity Data Scientist Project 2 - Disaster Response Pipeline Project

### Table of Contents

1. [Introduction](#Introduction)
2. [Descriptions](#Descriptions)
3. [Requirement](#Requirement)
4. [Instructions](#instructions)

## Project Introduction: <a name="Introduction"></a>
> The goal of the project is to classify the disaster messages into categories. In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages. Through a web app, the user can input a new message and get classification results in several categories. The web app also display visualizations of the data.



## Project Descriptions: <a name = "descriptions"></a>
The project has three componants which are:

1. **ETL Pipeline:** `process_data.py` file contain the script to create ETL pipline which:

- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. **ML Pipeline:** `train_classifier.py` file contain the script to create ML pipline which:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. **Flask Web App:** the web app enables the user to enter a disaster message, and then view the categories of the message.

The web app also contains some visualizations that describe the data. 
 

## Project Requirement: <a name = "Requirement"></a>
we will need an installation of Python 3, plus the following libraries:
* flask 
* jupyter
* nltk
* notebook
* numpy
* pandas
* plotly
* requests
* scikit-learn
* sqlalchemy

Run the following commands in the project's root directory to install the Requirement above :

`pip install -r requirements.txt`

## Project Instructions: <a name = "Instructions"></a>

To execute the app follow the instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Authors

* **Mohamed BOUSETTA MAHJOUB** - *Initial work* - [MedMahj](https://github.com/MedMahj/)
