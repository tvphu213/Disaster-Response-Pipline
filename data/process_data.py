import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    to load data from csv file
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df

def binary_convert(value, count):
    """
    """
    if value not in [0,1]:
        cnt+= 1 
    return value


def clean_data(df):
    """
    to clean data
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[1].values
    # use this row to extract a list of new column names for categories.
    category_colnames = [category[:-2] for category in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string and convert column from string to numeric
        categories[column] = [int(cell[-1:]) for cell in categories[column]]

    #binary value check and filter
    #check distribute of value and value counts
    for column in categories:
        print(column)
        print(categories[column].value_counts())
    
    #filter out invalid values
    for column in categories:
        categories = categories[categories[column].isin([0,1])]

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicate
    df.drop_duplicates(subset=['id'], inplace=True)
    return df



def save_data(df, database_filename):
    """to save data of df to database"""
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('MessageDetail', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
