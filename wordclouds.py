import numpy as np
import string, operator, gensim
from gensim.corpora import BleiCorpus
from gensim import corpora
from gensim.models import LdaMulticore
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

# Path to the dictionary and corpus
folder = 'models_unigram_topics_all_pos'
dictionary_path = folder+"/dictionary.dict"
corpus_path = folder+"/corpus.lda-c"
lda_num_topics = 250
lda_model_path = folder+"/lda_model_"+str(lda_num_topics)+"_topics.lda"

# Load using gensim
dictionary = corpora.Dictionary.load(dictionary_path)
corpus = corpora.BleiCorpus(corpus_path)
lda = LdaMulticore.load(lda_model_path)

# Mask for drawing the word cloud
custom_mask = np.array(Image.open("cloud.png"))

def drawWordCloud(wordList, topic_num):
	# wordList is a list of tuples (word, prob)
	# Multiplying the probability by 1000 and repeating the word since WordCloud takes in the text and not probability
	corpusList = [[elem[0]]*int(elem[1]*1000) for elem in wordList]
	corpusData = u" ".join([item for sublist in corpusList for item in sublist])
	wordcloud = WordCloud(background_color="white", mask=custom_mask).generate(corpusData)
	# Open a plot of the generated image.
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.savefig("wordclouds/"+str(topic_num)+".png")

nwords_per_topic = 200 # Maximum number of words to be shown in wordcloud
for i in range(lda_num_topics):
    print i
    topic_terms = lda.get_topic_terms(i, topn=nwords_per_topic)
    word_list = [(dictionary[x], y) for (x, y) in topic_terms]
    drawWordCloud(word_list, i)
