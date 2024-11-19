import os
from json import JSONEncoder

# pip install httpagentparser
import httpagentparser  # for getting the user agent as json
import nltk
from flask import Flask, render_template, session, request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc, SearchData, SessionData, DocData, ProcessData
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.core.utils import merge_dictionaries, preprocess_and_generate_wordcloud

import atexit
import pickle
import time
import pandas as pd


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

# end lines ***for using method to_json in objects ***

# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = 'afgsreg86sr897b4st8c79va8dy76fcs6g9d7'
# open browser dev tool to see the cookies
app.session_cookie_name = 'IRWA_SEARCH_ENGINE'

# instantiate our search engine
search_engine = SearchEngine()

# instantiate our in memory persistence
analytics_data = AnalyticsData()

# print("current dir", os.getcwd() + "\n")
# print("__file__", __file__ + "\n")
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
# print(path + ' --> ' + filename + "\n")
# load documents corpus into memory.
file_path = path + "/tweets-data-who.json"

# file_path = "../../tweets-data-who.json"
corpus = load_corpus(file_path)
print("loaded corpus. first elem:", list(corpus.values())[0])

start_time = time.time()
session_data = SessionData()
search_data = SearchData()
document_data = DocData()

def save_session_data():
    # save previous session data into pickle dictionary
    try: 
        with open('session_data.pickle', 'rb') as handle:
            session_register = pickle.load(handle)
    except FileNotFoundError:
        session_register = {}
    session_register[session_data.session_id] = session_data.to_dict()   
    
    with open('session_data.pickle', 'wb') as handle:
        pickle.dump(session_register, handle, protocol=pickle.HIGHEST_PROTOCOL)

# cleanup functions
def save_search_data():
    # save previous search data into pickle dictionary
    try: 
        with open('session_searches.pickle', 'rb') as handle:
            session_searches = pickle.load(handle)
    except FileNotFoundError:
        session_searches = {}
    if search_data.search_id!=-1:
        search_data.compute_total_time_spent()
        session_searches[search_data.search_id] = search_data.to_dict()
    
    with open('session_searches.pickle', 'wb') as handle:
        pickle.dump(session_searches, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_clicks():
    try:
        with open('session_clicks.pickle', 'rb') as handle:
            clicks = pickle.load(handle)
        clicks[session_data.session_id] = analytics_data.fact_clicks
        with open('session_clicks.pickle', 'wb') as handle:
            pickle.dump(clicks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        with open('session_clicks.pickle', 'wb') as handle:
            pickle.dump({session_data.session_id: analytics_data.fact_clicks}, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Function to be executed before the Flask app is closed: save into our database our analytics data
def cleanup_before_exit():
    save_session_data()
    save_search_data()
    save_clicks()
    check_left_doc()

# Register the cleanup function with atexit
atexit.register(cleanup_before_exit)

def save_doc_data():
    # save previous doc data into pickle dictionary
    try: 
        with open('docs_opened.pickle', 'rb') as handle:
            docs_opened = pickle.load(handle)
    except FileNotFoundError:
        docs_opened = {}
    docs_opened[document_data.doc_id] = document_data.to_dict()
    
    with open('docs_opened.pickle', 'wb') as handle:
        pickle.dump(docs_opened, handle, protocol=pickle.HIGHEST_PROTOCOL)

def check_left_doc():
    print("checking if left doc...")
    if document_data.just_seen:
        print("Yes, left doc")
        document_data.compute_total_time_spent()
        save_doc_data()
        document_data.set_as_not_seen()
    
    

# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")
    #start time only fisrt time

    session['start_time'] = start_time
    session['some_var'] = 'IRWA 2023 home'
    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    session_data.add_IP(user_ip)
    session_data.add_agent(agent)
    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))

    print(session)
    
    return render_template('index.html', page_title="Welcome")

def initialize_session_keys():
    session['session_id'] = session_data.session_id
    try:
        session['last_click_id']
    except KeyError:
        session['last_click_id'] = 0


@app.route('/search', methods=['POST'])
def search_form_post():
    check_left_doc()
    #start time
    time_search = time.time()
    global search_data

    search_query = request.form['search-query']
    algorithm=request.form['option']
    top_k = int(request.form['Top_k'])
    if not session_data.has_searched:
        initialize_session_keys()
    
    search_id = analytics_data.generate_search_id()
    print("search id: ", search_id)
    if algorithm=='option1':
        results = search_engine.search(search_query, search_id, corpus, top_k)
    if algorithm=='option2':
        results = search_engine.search_tf_idf(search_query, search_id, corpus, top_k)
    if algorithm=='option3':
        results = search_engine.search_boleans(search_query, search_id, corpus, top_k)

    

    print(search_query, algorithm, top_k)
    print(search_data.search_query, search_data.algorithm, search_data.top_k)
    if search_query != search_data.search_query or algorithm != search_data.algorithm or top_k != search_data.top_k:
        print("new search!!")
        if session_data.has_searched == 1:
            print("saving search data")
            save_search_data()
        else:
            session_data.search_session()

        found_count = len(results)
        search_lag=time.time()-time_search

        search_data = SearchData(search_id, session_data.session_id, search_query, algorithm, top_k, found_count, search_lag)


    return render_template('results.html', results_list=results, page_title="Results", 
                           found_counter=search_data.count, time_search=search_data.search_lag)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    # getting request parameters:
    # user = request.args.get('user')
    global document_data

    print("doc details session: ")
    print(session)

    res = session["some_var"]

    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args["id"]
    p1 = int(request.args["search_id"])  # transform to Integer
    p2 = int(request.args["param2"])  # transform to Integer
    print("click in id={}".format(clicked_doc_id))

    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))
    doc_details=corpus[int(clicked_doc_id)]

    session['last_click_id'] = (clicked_doc_id)

    document_data = DocData(clicked_doc_id, search_id=search_data.search_id, just_seen=True, doc=doc_details)
    document_data.compute_sentiment()
    return render_template('doc_details.html', search_data=search_data, doc_details = doc_details)


@app.route('/stats', methods=['GET'])
def stats():
    check_left_doc()
    """
    Show simple statistics example. ### Replace with dashboard ###
    :return:
    """
    #print time until now 
  
    
    docs = []
    # ### Start replace with your code ###

    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[int(doc_id)]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(row.id, row.title, row.description, row.doc_date, row.url, count)
        docs.append(doc)

    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs,time=time.time()-start_time)
    # ### End replace with your code ###



@app.route('/dashboard', methods=['GET'])
def dashboard():
    check_left_doc()
    process_data = ProcessData(session_data, search_data)
    process_data.pandify_all()
    process_data.compute_all_insights()
    visited_docs = []
    with open('session_clicks.pickle', 'rb') as handle:
        clicks = pickle.load(handle)
    # concatenate clicks and analytics_data.fact_clicks
    total_clicks = {}
    for click_session in clicks.values(): 
        total_clicks = merge_dictionaries(total_clicks, click_session)
    total_clicks = merge_dictionaries(total_clicks, analytics_data.fact_clicks)
    for doc_id in total_clicks.keys():
        d: Document = corpus[int(doc_id)]
        doc = ClickedDoc(doc_id, d.description, total_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)
    preprocess_and_generate_wordcloud(process_data.searches_df, 'search_query', 'wordcloud.png')
    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs,analytics_data=analytics_data, session_data=session_data)


@app.route('/sentiment')
def sentiment_form():
    check_left_doc()
    try:
        details=corpus[int(session['last_click_id'])]
        title=details.title
    except KeyError:
        title=''
    return render_template('sentiment.html',title=title)


@app.route('/sentiment', methods=['POST'])
def sentiment_form_post():
    check_left_doc()
    text = request.form['text']
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    score = ((sid.polarity_scores(str(text)))['compound'])
    return render_template('sentiment.html', score=score)


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=False)