import numpy as np
import string, operator, pickle, nltk, gensim
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import BleiCorpus
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import GradientBoostingRegressor
from collections import defaultdict

# Load the restaurant review data
restaurant_data_train = pickle.load(open("Pickles/restaurant_data_train.p","rb"))
restaurant_data_test = pickle.load(open("Pickles/restaurant_data_test.p", "rb"))
restaurant_reviews_train = pickle.load(open("Pickles/review_data_train.p", "rb"))
restaurant_reviews_test = pickle.load(open("Pickles/review_data_test.p","rb"))

# Create a dictionary with restaurant ID as key and the star rating as value for fast access
# Create set of restaurant IDs for test and train
restaurant_star_dict = defaultdict(float)
restaurant_id_train = set()
restaurant_id_test = set()
for l in restaurant_data_train:
    restaurant_star_dict[l['business_id']] = l['stars']
    restaurant_id_train.add(l['business_id'])

for l in restaurant_data_test:
    restaurant_star_dict[l['business_id']] = l['stars']
    restaurant_id_test.add(l['business_id'])

# Create a dictionary with restaurant ID as key and the concatenated list of reviews for that restaurant as value
review_text_train = defaultdict(unicode)
review_text_test = defaultdict(unicode)
# Iterate over training reviews
for l in restaurant_reviews_train:
    review_text_train[l['business_id']] = u" ".join((review_text_train[l['business_id']],unicode(l["text"].lower(), "utf-8")))

# Iterate over testing reviews
for l in restaurant_reviews_test:
    review_text_test[l['business_id']] = u" ".join((review_text_test[l['business_id']],unicode(l["text"].lower(), "utf-8")))

# Iterate over the dictionary to create the lists for the concatenated reviews, business ID, and star rating
X_train = []
y_train = []
bid_train = []
for bid, reviews_text in review_text_train.iteritems():
    y_train.append(restaurant_star_dict[bid])
    X_train.append(reviews_text)
    bid_train.append(bid)

X_test = []
y_test = []
bid_test = []
for bid, reviews_text in review_text_test.iteritems():
    y_test.append(restaurant_star_dict[bid])
    X_test.append(reviews_text)
    bid_test.append(bid)

# Make Bag-Of-Words features from the training data
trainBowModel = CountVectorizer(max_features=2000, stop_words='english')
train_bow_feats = trainBowModel.fit_transform(X_train)
trainVocab = trainBowModel.vocabulary_

# Get Bag-Of-Words features from the testing data using the model from training 
testBowModel = CountVectorizer(vocabulary=trainVocab, stop_words='english')
test_bow_feats = testBowModel.transform(X_test)

# Make TF-IDF features from the training data
trainTfiModel = TfidfTransformer()
train_tfi_feats = trainTfiModel.fit_transform(train_bow_feats)

# Get TF-IDF features from the testing data using the model from training 
test_tfi_feats = trainTfiModel.transform(test_bow_feats)

# Convert sparse matrix to numpy array
train_bow_feats = train_bow_feats.toarray()
test_bow_feats = test_bow_feats.toarray()
train_tfi_feats = train_tfi_feats.toarray()
test_tfi_feats = test_tfi_feats.toarray()

# Build a Gradient Boosting Regressor for predicting the star rating -- using BoW features
gbModel = GradientBoostingRegressor(n_estimators=200)
gbModel.fit(train_bow_feats,y_train)
y_pred = gbModel.predict(test_bow_feats)
mse = mean_squared_error(y_test[:len(y_pred)], y_pred)
print "MSE on BoW features:", mse

# Build a Gradient Boosting Regressor for predicting the star rating -- using TF-IDF features
gbModel = GradientBoostingRegressor(n_estimators=200)
gbModel.fit(train_tfi_feats,y_train)
y_pred = gbModel.predict(test_tfi_feats)
mse = mean_squared_error(y_test[:len(y_pred)], y_pred)
print "MSE on TF-IDF features:", mse

# Save the data into pickles
folder = 'tfidf'
pickle.dump(train_bow_feats, open(folder+"/train_bow_feats.p","wb"))
pickle.dump(test_bow_feats, open(folder+"/test_bow_feats.p","wb"))

pickle.dump(train_tfi_feats, open(folder+"/train_tfi_feats.p","wb"))
pickle.dump(test_tfi_feats, open(folder+"/test_tfi_feats.p","wb"))

pickle.dump(y_train, open(folder+"/y_train.p","wb"))
pickle.dump(y_test, open(folder+"/y_test.p","wb"))

pickle.dump(bid_train, open(folder+"/bid_train.p","wb"))
pickle.dump(bid_test, open(folder+"/bid_test.p","wb"))
