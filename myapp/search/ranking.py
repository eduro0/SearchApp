
"""This class needs to incorporate the ranking functionality. Leave the option open to evaluate performance and choose ranking algorithm"""


result = []

def run():
   pass

class ranking_tf_idf:

  result = []

  def __init__(self) -> None:
     pass

  def create_index_tfidf(lines, num_documents):
      """
      Implement the inverted index and compute tf, df and idf

      Argument:
      lines -- collection of tweets
      num_documents -- total number of documents

      Returns:
      index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
      list of tweets these keys appears in (and the positions) as values.
      tf - normalized term frequency for each term in each document
      df - number of documents each term appear in
      idf - inverse document frequency of each term
      """
      index = defaultdict(list)
      tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
      df = defaultdict(int)  # document frequencies of terms in the corpus
      idf = defaultdict(float)
      for tweet_id, line in lines.items():
          line = line.split('||')
          doc_number = id_to_doc_num[tweet_id]
          terms = build_terms(line[0] + line[2]) #text+?? hastag
          current_page_index = {}
          for position, term in enumerate(terms): # terms contains page_title + page_text. Loop over all terms
              try:
                  # if the term is already in the index for the current page (current_page_index)
                  # append the position to the corresponding list
                  current_page_index[term][1].append(position)
              except:
                  # Add the new term as dict key and initialize the array of positions and add the position
                  current_page_index[term]=[doc_number, array('I',[position])] #'I' indicates unsigned int (int in Python)

          #normalize term frequencies
          # Compute the denominator to normalize term frequencies (formula 2 above)
          # norm is the same for all terms of a document.
          norm = 0
          for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document.
            # posting ==> [current_doc, [list of positions]]
            # you can use it to infer the frequency of current term.
            norm += len(posting[1]) ** 2
          norm = math.sqrt(norm)
              # calculate the tf(dividing the term frequency by the above computed norm) and df weights
          for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm,4)) ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term] +=1 # increment DF for current term

        #merge the current page index with the main index
          for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

          for term in df:
            idf[term] = np.round(np.log(float(num_documents/df[term])), 4)


      return index, tf, df, idf

  """ example 
  num_documents = len(tweet_info)
  index, tf, df, idf = create_index_tfidf(tweet_info, num_documents)
  """

  """ 
  STARTING WITH TF_IDF SCORE 
  docs=search_docs_intersection(query,index)
  query = build_terms(query)
  ranked_docs = rank_documents(query, docs, index, idf, tf)
  top = 20 
  """

  #index, tf, df, idf = create_index_tfidf(lines, num_documents):

  #docs = search_docs_intersection(search_query,index)


  def query_processing(search_query):

    #preprocess
    def build_terms(line, tokenize=True):
      stemmer = PorterStemmer()
      stop_words = set(stopwords.words("english"))

      line = preprocess(line)

      line=  nltk.tokenize.word_tokenize(line) ## Tokenize the text to get a list of terms
      #here we have to decide if remove # or not and explain why
      line=[word for word in line if re.match(r'^\w+$', word)] ## remove punctuation
      line=[word for word in line if word not in stop_words]  ##eliminate the stopwords
      line=[nltk.stem.SnowballStemmer('english').stem(word) for word in line] ## perform stemming )

      if tokenize:
          return line
      else:
          return ' '.join(line)

      query = build_terms()
    return build_terms(search_query)


  # rank the documents with 
  # results_list = rank_documents(query_terms, docs, index, idf, tf):



class ranking_own_score:
   
   def __init__(self) -> None:
      pass