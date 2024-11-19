import datetime
import json
from random import random
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt
from myapp.search.algorithms import build_terms

from faker import Faker

fake = Faker()


# fake.date_between(start_date='today', end_date='+30d')
# fake.date_time_between(start_date='-30d', end_date='now')
#
# # Or if you need a more specific date boundaries, provide the start
# # and end dates explicitly.
# start_date = datetime.date(year=2015, month=1, day=1)
# fake.date_between(start_date=start_date, end_date='+30y')

def get_random_date():
    """Generate a random datetime between `start` and `end`"""
    return fake.date_time_between(start_date='-30d', end_date='now')


def get_random_date_in(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + datetime.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())), )


def load_json_file(path):
    """Load JSON content from file in 'path'

    Parameters:
    path (string): the file path

    Returns:
    JSON: a JSON object
    """

    # Load the file into a unique string
    with open(path) as fp:
        lines = fp.readlines()
    lines = [l.strip().replace(' +', ' ') for l in lines]
    datos_diccionario = [json.loads(line) for line in lines]
    return datos_diccionario

def merge_dictionaries(dict1, dict2):
    """Merge two dictionaries summing their commmon keys

    Parameters:
    dict1 (dict): the first dictionary
    dict2 (dict): the second dictionary

    Returns:
    dict: the merged dictionary
    """
    return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) | set(dict2)}


def preprocess_and_generate_wordcloud(dataframe, column_name, output_filename='output_images/wordcloud.png'):
    """
    Preprocesses the specified column in the DataFrame and generates a word cloud.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the text data.
    - column_name (str): The name of the column containing sentences.
    - output_filename (str): The name of the output image file (default is 'wordcloud.png').
    """
   
    

    # Apply preprocessing to the specified column
    dataframe[column_name] = dataframe[column_name].apply(build_terms)

    # Combine all preprocessed sentences into a single string
    all_sentences = ' '.join([' '.join(sublist) for sublist in dataframe[column_name].values])

    # Tokenize the preprocessed string
    tokens = word_tokenize(all_sentences)

    # Count the occurrences of each token
    word_counts = Counter(tokens)

    # Display word counts
    #print("Word Counts:")
    #print(word_counts)

    # Generate and save the word cloud
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(word_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off the axis labels
    plt.savefig(output_filename)
    plt.show()