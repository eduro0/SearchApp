import json
import random
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt

class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # statistics table 1
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = dict([])

    # statistics table 2
    fact_two = dict([])

    # statistics table 3
    fact_three = dict([])

    def generate_search_id(self) -> int:
        print(self)
        # return abs(hash(terms))
        return random.randint(0, 10000)


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
    

class SearchData:
    # create an initialization function that if the parameters are not passed, it will set the default values
    def __init__(self, search_id=-1, session_id=-1, search_query=-1, algorithm=0, top_k=0, count=0, search_lag=float(0)):
        self.search_id = search_id
        self.session_id = session_id
        self.start_time = time.time()
        self.search_query = search_query
        self.algorithm = algorithm
        self.top_k = top_k
        self.count = count
        self.search_lag = search_lag

    def compute_total_time_spent(self):
        return time.time() - self.start_time
    
    def to_dict(self):
        total_time_spent = self.compute_total_time_spent()
        return {
            "session_id": self.session_id,
            "search_query": self.search_query,
            "algorithm": self.algorithm,
            "top_k": self.top_k,
            "count": self.count,
            "total_time_spent": total_time_spent,
            "search_lag": self.search_lag
        }
    

class SessionData():
    def __init__(self, IP='', agent=''):
        self.session_id = -1
        self.set_session_id()
        self.has_searched = 0
        self.start_time = time.time()
        self.IP = IP
        self.agent = agent

    def add_IP(self, IP):
        self.IP = IP
    
    def add_agent(self, agent):
        self.agent = agent

    def search_session(self):
        self.has_searched = 1
    
    def compute_total_time_spent(self):
        return time.time() - self.start_time
    
    def set_session_id(self):
        try:
            with open('session_data.pickle', 'rb') as f:
                historic = pickle.load(f)
            self.session_id = len(historic)
        except FileNotFoundError:
            self.session_id = 0
    
    def compute_location(self):
        """
        Get the location of the user from the IP address
        Note : Should be used only on requets from the user because it slows down the process
        """
        response = requests.get('http://ip-api.com/json/'+ self.IP).json()
        print("GEODATA: ", response)

        if response['status'] is not None and response['country'] is not None and response['city'] is not None:
            if response['status'] == 'fail':
                self.location = 'Unknown'
            else :
                self.location = response['city'] + ', ' + response['country']
                return response['city'] + ', ' + response['country']

    def to_dict(self):
        total_time_spent = self.compute_total_time_spent()
        return {
            "total_time_spent": total_time_spent,
            "IP": self.IP,
            "agent": self.agent
        }


class DocData():
    def __init__(self, doc_id=-1, search_id=-1, just_seen=False, doc=None):
        self.doc_id = doc_id
        self.search_id = search_id
        self.just_seen = just_seen
        self.start_time = time.time()
        self.total_time_spent = 0
        self.doc = doc
        self.sentiment = None
        
    def compute_total_time_spent(self):
        return time.time() - self.start_time
    
    def set_as_not_seen(self):
        self.just_seen = False
    
    def compute_sentiment(self):
        nltk.download('vader_lexicon')
        sid = SentimentIntensityAnalyzer()
        score = ((sid.polarity_scores(str(self.doc.description)))['compound'])
        if score > 0:
            self.sentiment = 'positive'
        elif score < 0:
            self.sentiment = 'negative'
        else:
            self.sentiment = 'neutral'
    
    def to_dict(self):
        total_time_spent = self.compute_total_time_spent()
        return {
            "search_id": self.search_id,
            "total_time_spent": total_time_spent,
            "document_object": self.doc.to_json(),
            "sentiment": self.sentiment
        }

class ProcessData():
    def __init__(self, current_session, current_search):
        self.has_searched = current_session.has_searched
        with open('session_data.pickle', 'rb') as f:
            self.session_df = pickle.load(f) 
        if self.has_searched:
            self.session_df.update({current_session.session_id: current_session.to_dict()})

        with open('session_searches.pickle', 'rb') as f:
            self.searches_df = pickle.load(f)
        if self.has_searched:
            self.searches_df.update({current_search.search_id: current_search.to_dict()})

        with open('docs_opened.pickle', 'rb') as f:
            self.documents_df = pickle.load(f)

    def pandify_documents(self):
        # {'1575757996663377920': {'search_id': 6434, 'total_time_spent': 8.512597560882568, 'document_object': {'id': 1575757996663377920, 'title': '30.09.22.\nZaporizhzhia, Ukraine. \n\nThose people are volunteers who wanted to give some humanitarian ', 'description': '30.09.22.\nZaporizhzhia, Ukraine. \n\nThose people are volunteers who wanted to give some humanitarian aid for people there and leave town with their relatives. But russia had another plan. \n\n#russiaisateroriststate #RussiaIsANaziState #Zaporizhzhia #Ukraine #UkraineRussiaWar #war https://t.co/KAphafHmGC', 'doc_date': 'Fri Sep 30 08:02:37 0000 2022', 'likes': 1, 'retweets': 0, 'url': '', 'hashtags': ['russiaisateroriststate', 'RussiaIsANaziState', 'Zaporizhzhia', 'Ukraine', 'UkraineRussiaWar', 'war']}, 'sentiment': 'negative'}, ...
        df = pd.DataFrame(columns=['doc_id', 'search_id', 'total_time', 'description', 'doc_date', 'likes', 'retweets', 'hashtags', 'sentiment'])
        for doc_id, document_info in self.documents_df.items():
            df = pd.concat([df, pd.DataFrame([[doc_id, document_info['search_id'], document_info['total_time_spent'], document_info['document_object']['description'], document_info['document_object']['doc_date'], document_info['document_object']['likes'], document_info['document_object']['retweets'], document_info['document_object']['hashtags'], document_info['sentiment']]], columns=['doc_id', 'search_id', 'total_time', 'description', 'doc_date', 'likes', 'retweets', 'hashtags', 'sentiment'])])
        df.reset_index(inplace=True, drop=True)
        self.documents_df = df

    def pandify_searches(self):
        # {6434: {'session_id': 0, 'search_query': 'ukraine town', 'algorithm': 'option1', 'top_k': 50, 'count': 8, 'total_time_spent': 11.957694292068481, 'search_lag': 0.4843909740447998}, ..
        df = pd.DataFrame(columns=['search_id', 'session_id', 'search_query', 'algorithm', 'top_k', 'count', 'total_time_spent', 'search_lag'])
        for search_id, search_info in self.searches_df.items():
            df = pd.concat([df, pd.DataFrame([[search_id, search_info['session_id'], search_info['search_query'], search_info['algorithm'], search_info['top_k'], search_info['count'], search_info['total_time_spent'], search_info['search_lag']]], columns=['search_id', 'session_id', 'search_query', 'algorithm', 'top_k', 'count', 'total_time_spent', 'search_lag'])])
        df.reset_index(inplace=True, drop=True)
        self.searches_df = df
        return df

    def pandify_session(self):
        # {0: {'total_time_spent': 61.897597551345825, 'IP': '127.0.0.1', 'agent': {'platform': {'name': 'Windows', 'version': '10'}, 'os': {'name': 'Windows', 'version': '10'}, 'bot': False, 'browser': {'name': 'Chrome', 'version': '119.0.0.0'}}}}
        df = pd.DataFrame(columns=['session_id', 'total_time_spent', 'IP', 'platform_name', 'platform_version', 'os_name', 'os_version', 'bot', 'browser_name', 'browser_version'])
        for session_id, session in self.session_df.items():
            if session['agent'] != '' and session['IP'] != '':
                df = pd.concat([df, pd.DataFrame([[session_id, session['total_time_spent'], session['IP'], session['agent']['platform']['name'], session['agent']['platform']['version'], session['agent']['os']['name'], session['agent']['os']['version'], session['agent']['bot'], session['agent']['browser']['name'], session['agent']['browser']['version']]], columns=['session_id', 'total_time_spent', 'IP', 'platform_name', 'platform_version', 'os_name', 'os_version', 'bot', 'browser_name', 'browser_version'])])
            else:
                df = pd.concat([df, pd.DataFrame([[session_id, session['total_time_spent'], None, None, None, None, None, None, None, None]], columns=['session_id', 'total_time_spent', 'IP', 'platform_name', 'platform_version', 'os_name', 'os_version', 'bot', 'browser_name', 'browser_version'])])
        df.reset_index(inplace=True, drop=True)
        self.session_df = df
        return df
    
    def pandify_all(self):
        self.pandify_documents()
        self.pandify_searches()
        self.pandify_session()

    def searches_per_session_hist(self):
        data = self.searches_df.groupby('session_id').count()['search_id']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data, bins=20)
        ax.set_xlabel('Number of searches')
        ax.set_ylabel('Number of sessions')
        ax.set_title('Histogram of number of searches per session')
        fig.savefig('output_images/searches_per_session.png')
        
    def time_per_session_hist(self):
        data = self.session_df['total_time_spent']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data, bins=20)
        ax.set_xlabel('Time spent in seconds')
        ax.set_ylabel('Number of sessions')
        ax.set_title('Histogram of time spent per session')
        fig.savefig('output_images/time_per_session.png')

    def documents_seen_per_search(self):
        data = self.documents_df.groupby('search_id').count()['doc_id']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data, bins=20)
        ax.set_xlabel('Number of documents seen')
        ax.set_ylabel('Number of searches')
        ax.set_title('Histogram of number of documents seen per search')
        fig.savefig('output_images/documents_seen_per_search.png')
    
    def time_per_search(self):
        data = self.searches_df['total_time_spent']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data, bins=20)
        ax.set_xlabel('Time spent in seconds')
        ax.set_ylabel('Number of searches')
        ax.set_title('Histogram of time spent per search')
        fig.savefig('output_images/time_per_search.png')
    
    def compute_all_insights(self):
        self.searches_per_session_hist()
        self.time_per_session_hist()
        self.documents_seen_per_search()
        self.time_per_search()
        print("All insights computed and saved in output_images folder")