import random

from myapp.search.objects import ResultItem, Document
from myapp.search.algorithms import search_in_corpus, search_in_corpus_idf_itf,search_in_bolean


def build_demo_results(corpus: dict, search_id):
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    res = []
    size = len(corpus)
    ll = list(corpus.values())
    #for index in range(random.randint(0, 40)):
    #    item: Document = ll[random.randint(0, size)]
    #    res.append(ResultItem(item.id, item.title, item.description, item.doc_date,
    #                          "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), random.random()))

    for index, item in enumerate(corpus['Id']):
        # DF columns: 'Id' 'Tweet' 'Username' 'Date' 'Hashtags' 'Likes' 'Retweets' 'Url' 'Language'
        res.append(ResultItem(item.Id, item.Tweet, item.Tweet, item.Date,
                                "doc_details?id={}&search_id={}&param2=2".format(item.Id, search_id), random.random()))

    # simulate sort by ranking
    res.sort(key=lambda doc: doc.ranking, reverse=True)
    return res


class SearchEngine:
    """educational search engine"""

    def search(self, search_query, search_id, corpus,top_k):
        print("Search query:", search_query)

        results = []
        ##### your code here #####
        # results = build_demo_results(corpus, search_id)  # replace with call to search algorithm
        if search_query == '':
            return []
        ranked_tweets = search_in_corpus(corpus, search_query)
        ##### your code here #####
        if ranked_tweets == False:
            return []
        for index, item in enumerate(list(ranked_tweets)):
            if index >= top_k:
                break
        # DF columns: 'Id' 'Tweet' 'Username' 'Date' 'Hashtags' 'Likes' 'Retweets' 'Url' 'Language'
            results.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                                 "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), index))
        
        results.sort(key=lambda doc: doc.ranking, reverse=False)
        return results


    def search_tf_idf(self, search_query, search_id, corpus,top_k):
        print("Search query:", search_query)

        results = []
        ##### your code here #####
        # results = build_demo_results(corpus, search_id)  # replace with call to search algorithm
        if search_query == '':
            return []
        ranked_tweets = search_in_corpus_idf_itf(corpus, search_query)
        if ranked_tweets == False:
            return []
        ##### your code here #####
        for index, item in enumerate(list(ranked_tweets)):
            if index >= top_k:
                break
        # DF columns: 'Id' 'Tweet' 'Username' 'Date' 'Hashtags' 'Likes' 'Retweets' 'Url' 'Language'
            results.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                                 "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), index))
        
        results.sort(key=lambda doc: doc.ranking, reverse=False)
        return results
    
    def search_boleans(self, search_query, search_id, corpus,top_k):

        ##### your code here #####
        results = []
        # results = build_demo_results(corpus, search_id)  # replace with call to search algorithm
        if search_query == '':
            return []
        ranked_tweets = search_in_bolean(corpus, search_query)
        if ranked_tweets == False:
            return []
        
        for index, item in enumerate(list(ranked_tweets)):
            if index >= top_k:
                break
        # DF columns: 'Id' 'Tweet' 'Username' 'Date' 'Hashtags' 'Likes' 'Retweets' 'Url' 'Language'
            results.append(ResultItem(item.id, item.title, item.description, item.doc_date,
                                 "doc_details?id={}&search_id={}&param2=2".format(item.id, search_id), index))
        
        results.sort(key=lambda doc: doc.ranking, reverse=False)
        return results