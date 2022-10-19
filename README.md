## Disaster Response Pipeline Project

### Project Motivation
Disaster Response is a project in Udacity's data scientist course. 
Through this project we can understand the life cycle of data science and can do a small application of classification.
Specifically:
- Build etl pipeline
- Build machine learning pipeline
- Model evaluation and improvement using gridsearchcv

![Model evalue](/img/model_evalue.png")

- Build a web-app using the model to predict the category of the message as well as display analytical graphs using plotly

### File Description
app/templates/go.html -- list view of predicted categories from inputed message
app/templates/master.html -- master page
app/run.py -- route define
data/*.csv -- data set
data/process_data.py -- data process pipeline code file
data/DisasterResponse.db -- sqlite database save processed data 
models/classifier.pkl -- classification model
models/train_classifier.py -- machine learning pipline code file
img/* -- image for readme file

### Datasource
[Udacity](https://www.udacity.com/)


#### Acknowledgements
- Dataset and template files credit
[Udacity](https://www.udacity.com/)


### Instructions:

1. Create virtual environment and active it
    https://docs.python.org/3/library/venv.html
2. Install library
    pip install -r requirements.txt 
3. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Go to `app` directory: `cd app`

5. Run your web app: `python run.py`

6. Click the `PREVIEW` button to open the homepage

### What to do next?:

Looking at the chart, it's easy to see that we are using an imblanced dataset, there are categories that don't even have a sample. 
As a result, the model can have high accuracy, but the prediction in reality will not be correct. 
We can use sklearn's built-in library to resample and re-train and evaluate the model's accuracy.
[Reference url here ](https://elitedatascience.com/imbalanced-classes)
