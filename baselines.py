import numpy as np
import operator, pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict

# JSON parser
def parseJson(fname):
    for l in open(fname):
        try:
            yield eval(l)
        except:
            pass

# US state list -- used to remove restaurants not in the US
stateListUS = {"AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","GU","HI","IA","ID", "IL","IN","KS",\
"KY","LA","MA","MD","ME","MH","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY", "OH","OK",\
"OR","PA","PR","PW","RI","SC","SD","TN","TX","UT","VA","VI","VT","WA","WI","WV","WY"}

# Read data of all businesses
business_data = list(parseJson("yelp_academic_dataset_business.json"))

# Separate out restaurants from all the businesses
restaurant_data = []
for l in business_data:
    if 'Restaurants' in l['categories'] and  l['state'] in stateListUS:
        restaurant_data.append(l)
print "Number of restaurants:" len(restaurant_data)

# Split test and train data with 80:20 split
restaurant_data_train, restaurant_data_test = train_test_split(restaurant_data, test_size=0.20, random_state=42)

# Get the restaurant ID and star rating
restaurant_stars = []
restaurant_ids = []
for l in restaurant_data_train:
    restaurant_ids.append(l['business_id'])
    restaurant_stars.append(l['stars'])

restaurant_stars_test = []
restaurant_status_test = []
restaurant_ids_test = []
for l in restaurant_data_test:
    restaurant_ids_test.append(l['business_id'])
    restaurant_stars_test.append(l['stars'])

# Get average star rating from the training data
avg_stars = np.mean(restaurant_stars)

print "MSE using nation-wide average:", mean_squared_error(restaurant_stars_test, [np.mean(avg_stars)]*len(restaurant_stars_test))

# Slice the train data per city to find the average rating in a city
restaurant_city_train = defaultdict(list)
for l in restaurant_data_train:
    restaurant_city_train[l['city']].append(l['stars'])

# Calulate MSE on the test data
err = 0
for l in restaurant_data_test:
    city_list = restaurant_city_train[l['city']]
    if city_list == []:
        city_mean = 3.5547
    else:
        city_mean = np.mean(city_list)
    err += (l['stars'] - city_mean)**2.0
print "MSE using city-wise average:", err/len(restaurant_data_test)

# Get the average user rating for each restaurant -- should be close to the ground truth
restaurant_user_stars = defaultdict(list)
for l in restaurant_reviews_train:
    restaurant_user_stars[l["business_id"]].append(l["stars"])

# Calulate MSE on the train data -- no separate training need here
err=0.0
useravgList = []
trueStarList = []
for l in restaurant_data_train:
    user_rating_list = restaurant_user_stars[l["business_id"]]
    meanvalue = np.mean(user_rating_list)
    trueStarList.append(l["stars"])
    useravgList.append(meanvalue)
    err += (l["stars"]-round(meanvalue*2.0, 1) / 2.0)**2.0
print "MSE by averaging user-ratings:", err/len(restaurant_data_test)

# Get the different categories of restaurants
categoriesSet = defaultdict(int)
for nelem, l in enumerate(restaurant_data_train):
    for elem in l['categories']:
        categoriesSet[elem]=categoriesSet[elem]+1

# Print the categories sorted by count
print sorted(categoriesSet.items(), key=itemgetter(1),reverse=True)

# Read and save review data for topic modelling
review_data = list(parseJson("yelp_academic_dataset_review.json"))
restaurant_ids = set(restaurant_ids)
restaurant_ids_test = set(restaurant_ids_test)
restaurant_reviews_train = []
restaurant_reviews_test = []
for nelem, l in enumerate(review_data):
    if l['business_id'] in restaurant_ids:
        restaurant_reviews_train.append(l)
    elif l['business_id'] in restaurant_ids_test:
        restaurant_reviews_test.append(l)

# Number of reviews in training and testing
print "Number of reviews in training: ", len(restaurant_reviews_train)
print "Number of reviews in testing: ", len(restaurant_reviews_test)

# Save pickles
pickle.dump(restaurant_data_train, open("Pickles/restaurant_data_train.p","wb"))
pickle.dump(restaurant_data_test, open("Pickles/restaurant_data_test.p","wb"))

pickle.dump(restaurant_reviews_train, open("Pickles/review_data_train.p","wb"))
pickle.dump(restaurant_reviews_test, open("Pickles/review_data_test.p","wb"))