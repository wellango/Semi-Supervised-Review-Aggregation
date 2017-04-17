import nltk
import string
import operator
import gensim
import os
import pickle
import numpy as np
from collections import defaultdict
from gensim.corpora import BleiCorpus
from gensim import corpora
from gensim.models import LdaMulticore
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

restaurant_data_train = pickle.load(open("Pickles/restaurant_data_train.p","rb"))
restaurant_data_test = pickle.load(open("Pickles/restaurant_data_test.p", "rb"))
restaurant_reviews_train = pickle.load(open("Pickles/review_data_train.p", "rb"))
restaurant_reviews_test = pickle.load(open("Pickles/review_data_test.p","rb"))

print "loaded pickles"

restaurant_star_dict = defaultdict(float)
for l in restaurant_data_train:
    restaurant_star_dict[l['business_id']] = l['stars']

for l in restaurant_data_test:
    restaurant_star_dict[l['business_id']] = l['stars']

stopwords = {}
with open('stopwords.txt', 'rU') as f:
    for line in f:
        stopwords[line.strip()] = 1

lem = WordNetLemmatizer()
# Considering Nouns, Adjectives and Adverbs
posTagSet = set(["NN", "NNS", "JJ", "JJR", "JJS", "RB"])

corpus_list = []
nouns_dict = defaultdict(list)
for nelem, l in enumerate(restaurant_reviews_train):
    review_dict = {}
    review_dict['review_id'] = l['review_id']
    review_dict['user_id'] = l['user_id']
    review_dict['business_id'] = l['business_id']
    review_dict['text'] = l['text']
    nouns = []
    sentences = nltk.sent_tokenize(unicode(l["text"].lower(), "utf-8"))
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        text = [word for word in tokens if word not in stopwords]
        tagged_text = nltk.pos_tag(text)
        nouns += [lem.lemmatize(word) for word, tag in tagged_text if tag in posTagSet]
    review_dict['words'] = nouns
    nouns_dict[l['review_id']] = nouns
    corpus_list.append(review_dict)
    if nelem%1000==0:
        print "pos-tag", nelem


folder = 'models_unigram_topics_all_pos'
dictionary_path = folder+"/dictionary.dict"
corpus_path = folder+"/corpus.lda-c"
lda_num_topics = 250
lda_model_path = folder+"/lda_model_"+str(lda_num_topics)+"_topics.lda"
print "created paths"

dictionary = corpora.Dictionary(review["words"] for review in corpus_list)
dictionary.filter_extremes(keep_n=10000)
dictionary.compactify()
corpora.Dictionary.save(dictionary, dictionary_path)

corpus = [dictionary.doc2bow(review["words"]) for review in corpus_list]
BleiCorpus.serialize(corpus_path, corpus, id2word=dictionary)

corpus = corpora.BleiCorpus(corpus_path)
print "running lda"
lda = gensim.models.LdaMulticore(corpus, num_topics=lda_num_topics, id2word=dictionary, minimum_probability=0., workers=8)
lda.save(lda_model_path)
print "done lda"

dictionary = corpora.Dictionary.load(dictionary_path)
corpus = corpora.BleiCorpus(corpus_path)
lda = LdaMulticore.load(lda_model_path)
i = 0
for topic in lda.show_topics(lda_num_topics):
    print type(topic)
    print 'Topic #' + str(i) + ': ' + str(topic)
    i += 1


nwords_per_topic = 10
topic_words_set = set()
for i in range(lda_num_topics):
    topic_terms = lda.get_topic_terms(i, topn=nwords_per_topic)
    for term in topic_terms:
        topic_words_set.add(term[0])
# print 'Topic_Words = ', topic_words_set
print 'Topic_Words_len = ', len(topic_words_set)
topic_word_id = dict(zip(topic_words_set, range(len(topic_words_set))))

def displayTopics():
    dictionary = corpora.Dictionary.load(dictionary_path)
    corpus = corpora.BleiCorpus(corpus_path)
    lda = LdaMulticore.load(lda_model_path)
    i = 0
    for topic in lda.show_topics(lda_num_topics):
        print 'Topic #' + str(i) + ': ' + str(topic)
        i += 1

displayTopics()


class Predict():
    def __init__(self):
        self.dictionary = corpora.Dictionary.load(dictionary_path)
        self.lda = LdaMulticore.load(lda_model_path)
    def load_stopwords(self):
        stopwords = {}
        with open('stopwords.txt', 'rU') as f:
            for line in f:
                stopwords[line.strip()] = 1
        return stopwords
    def extract_lemmatized_nouns(self, new_review):
        stopwords = self.load_stopwords()
        words = []
        nouns = []
        sentences = nltk.sent_tokenize(unicode(l["text"].lower(), "utf-8"))
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            text = [word for word in tokens if word not in stopwords]
            tagged_text = nltk.pos_tag(text)
            nouns += [lem.lemmatize(word) for word, tag in tagged_text if tag in posTagSet] 
        return nouns
    def run(self, new_review, isTest):
        nouns = self.extract_lemmatized_nouns(new_review['text'])
        new_review_bow = self.dictionary.doc2bow(nouns)
        new_review_lda = self.lda[new_review_bow]
        return new_review_lda


# Training
print "Training"
restaurant_reviews_dict_train = defaultdict(list)
for l in restaurant_reviews_train:
    restaurant_reviews_dict_train[l['business_id']].append(l)

X_train = []
y_train = []
bid_train = []
predict = Predict()

nelem = 0
for bid, reviews_list in restaurant_reviews_dict_train.iteritems():
    y_train.append(restaurant_star_dict[bid])
    bid_train.append(bid)    
    feat = [0]*len(topic_words_set)
    for l in reviews_list:
        topic_predictions = predict.run(l, isTest=0)
        for topic_id, topic_prob in topic_predictions:
            topic_terms = lda.get_topic_terms(topic_id, topn=nwords_per_topic)
            for term in topic_terms:
                feat[topic_word_id[term[0]]] += term[1] * topic_prob
    X_train.append([x * 1.0 / len(reviews_list) for x in feat])
    nelem += 1
    if nelem%250 == 0:
        print "training", (nelem*100)/len(restaurant_reviews_dict_train), "%"


# Testing
print "Testing LR"
restaurant_reviews_dict_test = defaultdict(list)
for l in restaurant_reviews_test:
    restaurant_reviews_dict_test[l['business_id']].append(l)

X_test = []
y_test = []
bid_test = []

nelem = 0
for bid, reviews_list in restaurant_reviews_dict_test.iteritems():
    y_test.append(restaurant_star_dict[bid])
    bid_test.append(bid)
    feat = [0]*len(topic_words_set)
    for l in reviews_list:
        topic_predictions = predict.run(l, isTest=1)
        for topic_id, topic_prob in topic_predictions:
            topic_terms = lda.get_topic_terms(topic_id, topn=nwords_per_topic)
            for term in topic_terms:
                feat[topic_word_id[term[0]]] += term[1] * topic_prob
    X_test.append([x * 1.0 / len(reviews_list) for x in feat])
    nelem += 1
    if nelem%50 == 0:
        print "testing", (nelem*100)/len(restaurant_reviews_dict_test), "%"

folder = 'models_unigram_words_all_pos'

if not os.path.exists(folder):
    os.mkdir(folder)

pickle.dump(X_train, open(folder+"/X_train.p","wb"))
pickle.dump(X_test, open(folder+"/X_test.p","wb"))

pickle.dump(y_train, open(folder+"/y_train.p","wb"))
pickle.dump(y_test, open(folder+"/y_test.p","wb"))

pickle.dump(bid_train, open(folder+"/bid_train.p","wb"))
pickle.dump(bid_test, open(folder+"/bid_test.p","wb"))


gbModel = GradientBoostingRegressor(n_estimators=200)
gbModel.fit(X_train,y_train)
y_pred = gbModel.predict(X_test)
mse = mean_squared_error(y_test[:len(y_pred)], y_pred)
print "MSE  = ", mse
