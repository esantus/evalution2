# -*- coding: <coding> -*-

import os, sys, re, gzip, pickle, math
import numpy as np

import codecs
import traceback

import nltk
from nltk import word_tokenize
from nltk.stem import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.collocations import *
from nltk.probability import FreqDist


import panda as pd

from docopt import docopt
from collections import defaultdict
from composes.utils import io_utils
from composes.semantic_space.space import Space
from scipy.sparse import coo_matrix, csr_matrix
from composes.matrix.sparse_matrix import SparseMatrix



class CorpusManagement(corpus_filename):
	"""
	CorpusManagement can be used to manage raw and preprocessed linguistic corpora.

	Among the other features, it allows:
		- Corpus Preprocessing: convert a raw text file into a tsv file, containing
		the following columns: token, lemma, pos, index, parent, dep. More columns
		can be defined and added in the future.
		- Statistics Extraction: extract metadata about words in a wordlist.
		- Pattern Search: given a preprocessed corpus, patterns can be searched and
		extracted.

    Note:
        Each method is documented separately.

    Args:
        corpus_filename (string): filename of the corpus, either preprocessed or not

    Attributes:
        name (type): Description

    Methods:
    	__init__
    	preprocess_corpus
    	extract_statistics
    	pattern_search
    """



    CORPUS_FIELDS = 6
    """
    Number of fields in the tsv corpus datafile.
    """



    TAB_SEP = "\t"
    """
    List of constants spelling out the separators
    """



	TOKEN = 0
	LEMMA = 1
	POS = 2
	INDEX = 3
	PARENT = 4
	DEP = 5
	"""
	List of corpus fields
	"""   



	def __init__(self, corpus_filename):
		"""
		__init__ initializes the class, by loading the corpus and setting the attributes

		Args:
	        corpus_filename (string): filename of the corpus

	    Returns:
	        True for success, False otherwise.
	    """

	    try:
	    	# In case it is gz file
	    	if is_gz(corpus_filename):
	    		self.corpus = gzip.open(corpus_filename, 'r')
	    	else:
	    		self.corpus = open(corpus_filename, "r")

	    	# Check whether it needs to be processed
	    	self.to_preprocess = self._is_corpus_to_be_preprocessed(self.corpus)

	    	# If it needs to be processed
	    	if self.to_preprocess == True:
	    		pass

	    	# Otherwise
	    	else:
	    		pass

	    	# Initialize stopwords
	    	self.stopwords = set(stopwords.words('english'))

	    except:
	    	print("CorpusManagement, __init__: It was not possible to open the corpus: " + corpus_filename)
	    	return False



	def __is_gz(corpus_filename):
		"""
		is_gz returns True if the file has "gz" extension, otherwise it returns False

		Args:
			corpus_filename (stirng): name of the corpus file
		Returns:
			True if the file has "gz" extension, otherwise it returns False
		"""

		return True if corpus_filename.endswith("gz") else False



	def __is_corpus_to_be_preprocessed(self, corpus):
		"""
		_is_corpus_to_be_preprocessed returns True if the first line of the corpus cannot be split into
		CORPUS_FIELDS fields

		Args:

	    Returns:
	        True if the corpus needs to be processed, False otherwise.
	    """

		first_line = corpus.readline()

		if len(first_line.strip.split(TAB_SEP)) == CORPUS_FIELDS:
			return False
		else:
			return True



	def preprocess_corpus(self, raw_corpus_filename):
		"""
		preprocess_corpus convert a raw text file into a tsv file, containing
		the following columns: token, lemma, pos, index, parent, dep. More columns
		can be defined and added in the future.

		Args:
	        raw_corpus_filename (string): file to be opened and processed.

	    Returns:
	        True for success, False otherwise. It prints out the processed corpus
	        in a file with name: raw_corpus_filename + "_outoput.txt"
	    """
	    pass



	def extract_statistics(self, wordlist):
		"""
		extract_statistics extracts a number of statistical information from
		the corpus for each word in the provided wordlist, returning a dataframe.
		More columns information can be defined and added in the future.

		Args:
	        wordlist (set of strings): file to be opened and processed.

	    Returns:
	        The return value. True for success, False otherwise.
		"""
		pass



	def pattern_search(self, pattern):
		"""
		pattern_search is useful to find patterns in the corpus, such as those
		that link two words. this function returns the pattern as list of tuples,
		each of which containing all the csv fields in the corpus.

		Args:
	        pattern (strings): pattern, written as a csv string containing the
	        following elements: column=value, quantifiers (i.e., "*+?{}[]").

	    Returns:
	        It yields the file and offsets of where to find the pattern.
		"""
		pass



	def extract_ngrams(self, wordlist, win, stopwords=True, lemma=True, pos=True, dep=True, PLMI=False):
		"""
		extract_ngrams searches ngrams of size win (with or without stopwords)
		for all the words in the wordlist.

		Args:
	        wordlist (list of strings): list of words for which we want to extract
	        ngrams
	        win (int): number of words in the ngram
	        stopwords (bool): True if stopwords should be considered, False otherwise
	        lemma (bool): True if the lemmatized ngrams should be extracted, False if
	        the tokenized
	        pos (bool): True if the POS should be attached to the tokens/lemmas
	        dep (bool): True if the dep should be attached to the tokens/lemmas
	        PLMI (bool): True to assign PLMI score, False to assign PPMI

	    Returns:
	        It returns a dictionary of ngram sets, for every word in wordlist
		"""

		if stopwords == False:
			stpw_list = set(stopwords.words('english'))

		norm = raw_input("Do you want to normalize the PMI by ngram frequency? (y/n) ")

		word_freq = {}
		tot_word_freq = 0

		ngram_freq = {}
		tot_ngram_freq = 0

		colloc = {}

		for sentence in __get_sentences(self.corpus):

			sentence_fields = zip(*sentence)

			# Removing stopwords from the sentence
			if stopwords == False:
				sentence_fields = [w for w in sentence_fields if w not in self.stopwords]

			# Take the sentence as all lemmas
			if lemma == True:
				FIELD = LEMMA

			# Take the sentence as all tokens
			else:
				FIELD = TOKEN

			# Save the POS
			if pos == True and dep == False:
				sentence = sentence_fields[FIELD] + '-' + sentence_fields[POS]
			# Save both the dep and the POS
			elif pos == True and dep == True:
				sentence = sentence_fields[DEP] + ':' + sentence_fields[FIELD] + '-' + sentence_fields[POS]
			# Save only the token/lemma
			else:
				sentence = sentence_fields[FIELD]

			# Saving the frequency of every word
			for word in sentence:
				word_freq[word] = 0 if word not in word_freq else (word_freq[word]+1)
				tot_word_freq += 1

			# Generating all the ngrams
			ngrams = [tuple(sentence[i:i + win]) for i in range(len(sentence) - 1)]

			# Saving the frequency of every ngram
			for ngram in ngrams:
				ngram_freq[ngram] = 0 if ngram not in ngram_freq else (ngram_freq[ngram]+1)
				tot_ngram_freq += 1

		# For every ngram that was identified
		for ngram in ngram_freq:

			# Filtering only ngrams for the wordlist words
			

			# In calculating PPMI, put a cutoff of freq > 3 to avoid rare events to affect the rank
			if ngram_freq[ngram] > 3:
				ngram_prob = float(ngram_freq[ngram])/tot_ngram_freq

				# Initializing the variable to calculate the probability of components as independent events
				components_prob = 1
				for word in ngram:
					components_prob *= float(word_freq[word])/tot_word_freq

			# Ngram probability in PPMI
			colloc[ngram] = math.log(ngram_prob/components_prob) # PPMI

			# Adaptation to PLMI
			if PLMI == True:
				colloc[ngram] *= ngram_freq[ngram] #PLMI
		
		return colloc


	def __sort_ngram(colloc):
		return sorted([(ngram, colloc[ngram]) for ngram in colloc], key=lambda x:x[1], reverse=True)


	def __get_sentences(corpus):
    """
    _get_sentences returns all the sentences in a corpus file, yielding them one by one

    Args:
    	corpus (file): file containing the corpus
    
    Returns:
    	sentence (list of tuples): a list of tuples, each of which containing a token and all
    	the related data (i.e., token, lemma, pos, index, parent, dep)
    """

    s = []

    # For every line in corpus
    for line in corpus:
        line = line.decode('ISO-8859-2')

        # Ignore start and end of DOC
        if '<text' in line or '</text' in line or '<s>' in line:
            continue

        # Yield at the end of SENTENCE
        elif '</s>' in line:
            yield s
            s = []

        # Append all TOKENS
        else:

            try:
                word, lemma, pos, index, parent, dep = line.split()
                s.append((word, lemma, pos, int(index), int(parent), dep))

            # When one of the items is a space, ignore this TOKEN
            except:
                continue


