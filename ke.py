import numpy as np # linear algebra
import pandas as pd # data processing
df = pd.read_csv('papers.csv') #Read file

import re #Regular expression
from nltk.corpus import stopwords #Filter out words such as is, as, the, it, etc.,
from nltk.stem.wordnet import WordNetLemmatizer #Lemmatization of words Ex: Trusting to Trust

stop_words = set(stopwords.words('english')) #Load English stop words
new_words = ["fig","figure","image","sample","using", #Add a list of custom stopwords
             "show", "result", "large", 
             "also", "one", "two", "three", 
             "four", "five", "seven","eight","nine"]
stop_words = list(stop_words.union(new_words)) #Union English + custom stopwords

def pre_process(text): #pre-process text
    
    text=text.lower()     # convert text to lowercase
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)  #remove tags
    text=re.sub("(\\d|\\W)+"," ",text) # remove special characters and digits
    text = text.split() #Convert from string to list
    text = [word for word in text if word not in stop_words]     # Filter out stopwords
    text = [word for word in text if len(word) >= 3] # Filter out words less than three letters
    lmz = WordNetLemmatizer() # lemmatize
    text = [lmz.lemmatize(word) for word in text] #Lemmatize text
    return ' '.join(text)
docs = df['paper_text'].apply(lambda x:pre_process(x)) #anonymous function returning preprocessed text

from sklearn.feature_extraction.text import CountVectorizer #Count words in text
#create a vocabulary of words, 
cv=CountVectorizer(max_df=0.95,         # ignore words that appear in 95% of documents
                   max_features=10000,  # size of the vocabulary
                   ngram_range=(1,3)    # vocabulary contains single words, bigrams, trigrams
                  )
word_count_vector=cv.fit_transform(docs)
#print ("Word_count:",word_count_vector)

from sklearn.feature_extraction.text import TfidfTransformer #Term/Text Frequency Inverse Document Frequency. Term Frequency - provides more importance to the word that is more frequent in the document; Inverse Document Frequency- provides more weightage to the word that is rare in the corpus (all the documents).keywords are the words with the highest TF-IDF score

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
idf= tfidf_transformer.fit(word_count_vector)

def sort_coo(coo_matrix): #sparse matrix container that stores one row and column entry per nonzero
    tuples = zip(coo_matrix.col, coo_matrix.data) #Get column value and data value
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True) #sort in reverse order

def extract_topn_from_vector(feature_names, sorted_items, topn=10): #Get the feature names and tf-idf score of top n items
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx] #feature names in index/doc number
        
        score_vals.append(round(score, 3)) #keep track of feature name and its corresponding score
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

feature_names=cv.get_feature_names() # get feature names

def get_keywords(idx, docs):

    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs[idx]])) #generate tf-idf for the given document
    sorted_items=sort_coo(tf_idf_vector.tocoo()) #sort the tf-idf vectors by descending order of scores
    keywords=extract_topn_from_vector(feature_names,sorted_items,10) #extract only the top n; n here is 10
    
    return keywords

def print_results(idx,keywords, df): # Print results
    print("\n*******Title*******")
    print(df['title'][idx])
    print("\n**********Abstract Summary*******")
    print(df['abstract'][idx])
    print("\n*****Keywords*****")
    for k in keywords:
        print(k,keywords[k])
idx=941
keywords=get_keywords(idx, docs)
print_results(idx,keywords, df)