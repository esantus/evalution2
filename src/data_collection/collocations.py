import math
import nltk

from nltk import word_tokenize
from nltk.stem import *
from nltk.collocations import *
from nltk.probability import FreqDist
from nltk.corpus import stopwords

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

# Ngrams with 'creature' as a member
#creature_filter = lambda *w: 'League' not in w


while(True):
	try:
		window = int(raw_input("How many words in the ngram? "))
		break
	except:
		continue

while(True):
	try:
		norm = raw_input("Do you want to normalize the PMI by ngram frequency? (y/n) ")
		if norm in ["y", "n"]:
			break
	except:
		continue


while(True):
	try:
		stpw = raw_input("Do you want stop words to be removed? (y/n) ")
		if stpw in ["y", "n"]:
			if stpw == "y":
				stpw_list = set(stopwords.words('english'))
			break
	except:
		continue

filename = raw_input("Insert the name of the file you want to open: ")

with open("/Users/esantus/Projects/"+filename, "r") as f_in:

	word_freq = {}
	w_freq = 0
	ngram_freq = {}
	ngr_freq = 0
	colloc = {}

	text = f_in.read()

	lines = [l.strip() for l in text.split("\n")]

	for line in lines:
		
		if stpw == "y":
			words = [w for w in line.split(" ") if w not in stpw_list]
		else:
			words = line.split(" ")

		for word in words:
			if word not in word_freq:
				word_freq[word] = 0

			word_freq[word] += 1
			w_freq += 1


		ngrams = [tuple(words[i:i + window]) for i in range(len(words) - 1)]

		for ngram in ngrams:
			if ngram not in ngram_freq:
				ngram_freq[ngram] = 0

			ngram_freq[ngram] += 1
			ngr_freq += 1


	# A SOLUTION IS TO PUT A CUTOFF OF 3 AS MINIMUM FREQUWNCY TO AVOID RANK INFLUENCED BY RARE EVENTS
	for ngram in ngram_freq:

		#colloc[ngram] = 0 #INITIALIZING THE COLLOCATION

		ngram_prob = float(ngram_freq[ngram])/ngr_freq
		components_prob = 1

		for word in ngram:
			components_prob *= float(word_freq[word])/w_freq
			#colloc[ngram] += word_freq[word]  #PLMI NORMALIZATION BY THE FREQUENCY OF COMPONENT WORDS

		if norm == "y":
			colloc[ngram] = math.log(ngram_prob/components_prob) * ngram_freq[ngram] #PLMI MULTIPLICATION
		else:
			colloc[ngram] = math.log(ngram_prob/components_prob)


	ngrams = sorted([(ngram, colloc[ngram]) for ngram in colloc], key=lambda x:x[1], reverse=True)
	print(ngrams[:100])

	#print(word_freq)


	#print(nltk.FreqDist(text).most_common(20))

	#tokens = nltk.word_tokenize(text)
	#print(tokens[:20])

## Bigrams
#finder = BigramCollocationFinder.from_words(tokens, window_size=3)
# only bigrams that appear 3+ times
#finder.apply_freq_filter(3)
# only bigrams that contain 'creature'
#finder.apply_ngram_filter(creature_filter)
# return the 10 n-grams with the highest PMI
#print finder.nbest(bigram_measures.likelihood_ratio, 10)

"""
## Trigrams
finder = TrigramCollocationFinder.from_words(
   nltk.corpus.genesis.words('/Users/esantus/Projects/output_corpus.txt'))
# only trigrams that appear 3+ times
finder.apply_freq_filter(3)
# only trigrams that contain 'creature'
#finder.apply_ngram_filter(creature_filter)
# return the 10 n-grams with the highest PMI
print finder.nbest(trigram_measures.likelihood_ratio, 10)
"""
"""
with open("/Users/esantus/Projects/output_corpus.txt", "r") as f_in:
	text = f_in.read()
	raw_input("\n\n\nType something and we start tokenizing the text...")
	tokens = nltk.word_tokenize(text)
	print(tokens)
	print("\n\n\nText tokenized. Calculating n-grams.")
	tok_text = nltk.Text(tokens)
	tok_text.collocations()
"""
