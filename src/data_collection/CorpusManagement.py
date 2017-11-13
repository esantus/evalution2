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
from bs4 import BeautifulSoup

from docopt import docopt
from collections import defaultdict
from composes.utils import io_utils
from composes.semantic_space.space import Space
from scipy.sparse import coo_matrix, csr_matrix
from composes.matrix.sparse_matrix import SparseMatrix



# Setting the character type
reload(sys)
sys.setdefaultencoding("ISO-8859-1")



class CorpusManagement(corpus_filename):
	"""
	CorpusManagement can be used to manage raw and preprocessed linguistic corpora.
	It contains private and public functions for corpus management. Every object
	refers to a specific corpus, which can be in single or multiple files, in txt
	or gzip format.

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
        corpus_filenames (list of strings): filename(s) of the corpus, either
		preprocessed or not

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



	def __init__(self, corpus_filenames):
		"""
		__init__ initializes the class, by loading the corpus and setting the attributes

		Args:
	        corpus_filenames (list of strings): filename(s) of the corpus

	    Returns:
	        True for success, False otherwise.
	    """

		# Dictionary of the corpus files
		self.corpus = {}

		# Dictionary of the corpus files attributes
		self.to_preprocess = {}

		for corpus_filename in corpus_filenames:
		    try:
		    	# In case it is gz file
		    	if self.is_gz(corpus_filename):
		    		self.corpus[corpus_filename] = gzip.open(corpus_filename, 'r')
		    	else:
		    		self.corpus[corpus_filename] = open(corpus_filename, "r")

		    	# Check whether it needs to be processed
		    	self.to_preprocess[corpus_filename] = self.has_to_be_preprocessed(self.corpus[file_name])

		    except:
		    	print("CorpusManagement, __init__: It was not possible to open the corpus: " + corpus_filename)
		    	return False



	def return_stopwords(self):
		"""
		load_stopwords returns a set of stopwords

		Args:

		Returns:
			Set of stopwords
		"""

		return set(stopwords.words('english'))



	def is_gz(self, corpus_filename):
		"""
		is_gz returns True if the file has "gz" extension, False otherwise

		Args:
			corpus_filename (string): name of the corpus file
		Returns:
			True if the filename ends with "gz", False otherwise
		"""

		return True if corpus_filename.endswith("gz") else False



	def has_to_be_preprocessed(self, corpus):
		"""
		has_to_be_preprocessed returns True if the first line of the corpus cannot be split into
		CORPUS_FIELDS fields

		Args:

	    Returns:
	        True if the corpus needs to be processed, False otherwise.
	    """

		first_line = corpus.readline()

		if len(first_line.strip.split(TAB_SEP)) != CORPUS_FIELDS:
			return True
		else:
			return False



	def preprocess_corpus(self, raw_corpus):
		"""
		preprocess_corpus converts a raw text file into a tsv file, containing
		the following columns: token, lemma, pos, index, parent, dep. The output
		is saved in a file named "processed_corpus.txt". More columns can be
		defined and added in the future.

		Args:
	        raw_corpus (file): raw file to be processed

	    Returns:
	        True for success, False otherwise. It prints out the processed corpus
	        in a file with name: "processed_corpus.txt"
	    """

		# Output filename
		output_filename = "processed_corpus.txt"

		with codecs.open(output_filename, "w", "utf-8") as f_corpus:

			# Loading spacy's parser
			parser = English()

			# Sentences in a paragraph
			sentences = []

			for line in raw_corpus:

				# Cleaning the line from links and similar
				line = self.clean_line(line)

				# Ignore empty lines
				if line != "\n":
					parsed_line = parser(line)

	                # Sentence breaker
	                for span in parsed_line.sents:
						sentences.append([parsed_line[i] for i in range(span.start, span.end) if parsed_line[i] != "\n"])

					for i in range(0, len(sentences)):

						# BOS
						f_corpus.write("<s>\n")

						# Token, Lemma, POS, Index, Parent-Index, Dep
						for token in sentences[i]:
							f_corpus.write(token.orth_ + "\t" + token.lemma_.lower() + "\t" + token.pos_ + "\t" + str(token.i) + "\t" + (str(token.head.i) if token.head.i != 0 else "0") + "\t" + token.dep_ + "\n")

						# EOS
	                    f_corpus.write("</s>\n")



	def clean_line(self, line):
		"""
		clean_line returns a clean string, without xml or links

		Args:
			line (string): the string to be processed

		Returns:
			Clean string
		"""

		# Similar to what BeautifulSoup(line) does
		return self.remove_links(line.strip())



	def remove_links(self, line):
		"""
		remove_links removes links from a string

		Args:
			line (string): the string to be processed

		Returns:
			Clean string
		"""

		link_removal = re.compile(r'<[^>]+>')
 		return link_removal.sub('', line)



	def findWord(self, word, flags=re.IGNORECASE):
		"""
		findWord returns all instances of a word (i.e., MWE) in a sentence

		Args:
			word (string): word that need to be searched; it can be a MWE
			flags (re annotations): whether or not case CASE_SENSITIVE

		Returns:
			List of found words
		"""

		# Making sure that special characters are interpreted as simple characters
		w= w.replace(".", "\\.")
		w= w.replace("*", "\\*")
		W= w.replace("?", "\\?")

		# Returning the result of findall
		return re.compile(r'\b({0})\b'.format(w), flags).findall



	def whichCapit(self, word):
		"""
		whichCapit returns a string describing the capitalization type

		Args:
			word (string): the word to be analyzed

		Returns:
			string describing the capitalization type (e.g. none, first, all, others)
		"""

		# Return none if none is upper, all if none is lower, first if only first is upper, other otherwise
		return "none" if not any(map(str.isupper, str(word))) else "all" if not any(map(str.islower, str(word))) else "first" if ((str.isupper(str(word[0])) and not any(map(str.isupper, str(word[1:]))))) else "others"



	def extract_statistics(self, wordlist):
		"""
		extract_statistics extracts a number of statistical information from
		the corpus for each word in the provided wordlist, returning True when
		it succeeds. Statistics are saved in a file named "statistics.txt"
		More columns information can be defined and added in the future.

		Args:
	        wordlist (set of strings): file to be opened and processed.

	    Returns:
	        The return value. True for success, False otherwise.
		"""

		# Output file name
		output_filename = "statistics.txt"

		statistics = {}

		# Opening the output file
		with codecs.open(output_filename, "w", "utf-8") as f_statistics:

			# THE CODE HERE NEEDS TO IMPROVE TO ACCOUNT FOR MWE. IN PARTICULAR
			# WE NEED TO DO TO FIND A FAST WAY TO CHECK WHICH OF THE WORDS IN
			# WORDLIST MATCH THE SENTENCE AND FOR EACH OF THEM (SOME MAY OVERLAP)
			# INCREMENT THE FREQUENCY DICTIONARY, THE NORMALIZED DICTIONARY
			# THE CAPITALIZATION DICTIONARY AND THE POS-DEP DICTIONARY.

			# Convert the wordlist in a regular expression with lookahead (no consuming the input)
			#all_words_OR = '|'.join(re.escape(w) for w in wordlist)
			#wordlist_rex = re.compile(r'\b(?=(' + all_words_OR + r')(-\w:\w))\b')

			# Dictionary for extracting collocations
			ngram_win2_ppmi_slpd = {word_freq : {}, tot_word_freq : 0, ngram_freq : {}, tot_ngram_freq : 0, collocations : {}}
			ngram_win3_ppmi_slpd = {word_freq : {}, tot_word_freq : 0, ngram_freq : {}, tot_ngram_freq : 0, collocations : {}}

			# For every sentence
			for sentence in self.get_sentences(self.corpus):

				# Joining all tokens in the sentence
				#raw_sentence = " ".join(zip(*sentence)[0])

				# Removing spaces that may block matching (e.g. "state - of - the - art")
				#raw_sentence = raw_sentence.replace(" - ", "-")
				#raw_sentence = raw_sentence.replace(" , ", ", ")
				#raw_sentence = raw_sentence.replace(" . ", ".")

				#print(sentence)

				# For every word in wordlist that was found in the sentence
				#for word in wordlist_rex.findall:

				for token in sentence:

					if token[TOKEN] in wordlist:

						# word is the token[TOKEN]
						word = token[TOKEN]

						# Saving the record for word
						if word not in statitistcs:
							statistics[word]["freq"] = 0
							statistics[word]["norm"] = {}
							statistics[word]["capit"] = {"first":0, "all":0, "none":0, "others":0}
							statistics[word]["pos_dep"] = {}

						try:
							# Incrementing frequency
							statistics[word]["freq"] += 1

							# Saving inflected-lower-case terms and their frequency
							if match.lower() not in statistics[word]["norm"]:
								statistics[word]["norm"][match.lower()] = 0
							statistics[word]["norm"][match.lower()] += 1

							# Saving the capitalization type
							statistics[word]["not_norm"][self.whichCapit(match)] += 1

							# Saving POS-Dep combination
							if token[POS] + "_" + token[DEP] not in statistics[word]["pos_dep"]:
								statistics[word]["pos_dep"][token[POS] + "_" + token[DEP]] = 0
							statistics[word]["pos_dep"][token[POS] + "_" + token[DEP]] += 1

							ngram_win2_ppmi_slpd = update_collocations(sentence, ngram_win2_ppmi_slpd, wordlist, 2, stopwords=True, lemma=True, pos=True, dep=True, PLMI=False)
							ngram_win3_ppmi_slpd = update_collocations(sentence, ngram_win3_ppmi_slpd, wordlist, 3, stopwords=True, lemma=True, pos=True, dep=True, PLMI=False)

							# PATTERN: COOCCURRENCIES BETWEEN WORDS AND PATTERNS

						except:
							print ("Error in function extract_statistics: unable to update the statistics dictionary")
							continue


					# PRINT or DUMP AT EVERY TIME AND SAVE ALSO CORPUS POSITION, SO THAT YOU CAN RECOVER IN CASE OF UNEXPECTED QUIT

					#Collocations
						#Bigrams, Threegrams, Fourgrams
							#Dep:TargetToken-POS Dep:ContextToken-POS
							#Dep:TargetLemma-POS Dep:ContextLemma-POS




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
	        wordlist (set of strings): list of words for which we want to extract
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
			stpw_list = self.return_stopwords()

		for sentence in self.get_sentences(self.corpus):

			sentence_fields = zip(*sentence)

			# Removing stopwords from the sentence
			if stopwords == False:
				sentence_fields = [w for w in sentence_fields if w not in stpw_list]

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

			# Calculating the ngrams only for ngrams containing at least one word in wordlist
			if [word for word in set([w for w in ngram.split()]) if word in wordlist] != []:

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



def update_collocations(self, sentence, collocations, word_freq, tot_word_freq, ngram_freq, tot_ngram_freq, wordlist, win, stopwords=True, lemma=True, pos=True, dep=True, PLMI=False):
	"""
	update_collocations updates the collocations dictionary for every sentence that is passed to it.

	Args:
		sentence (list of tuples): each tuple contains token, lemma, pos, index,
		parent_index and dep
		collocations (dictionary): it contains the updated data
		wordlist (set of strings): list of words for which we want to extract
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

		# Removing stopwords from the sentence
		if stopwords == False:
			stpw_list = self.return_stopwords()
			sentence_fields = [w for w in sentence_fields if w not in stpw_list]

		# Turning sentence in list of fields
		sentence_fields = zip(*sentence)[0]

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

		# Calculating the ngrams only for ngrams containing at least one word in wordlist
		if [word for word in set([w for w in ngram.split()]) if word in wordlist] != []:

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

	return colloc, word_freq, tot_word_freq, ngram_freq, tot_ngram_freq



	def __sort_ngram(colloc):
		"""
		__sort_ngram return the sorted dictionary

		Args:
			colloc (dictionary of strings): dictionary containing the ngrams and their frequency

		Returns:
			colloc (dictionary): sorted dictionary
		"""

		return sorted([(ngram, colloc[ngram]) for ngram in colloc], key=lambda x:x[1], reverse=True)



	def get_sentences(corpus_dictionary):
	    """
	    _get_sentences returns all the sentences in a corpus file, yielding them one by one

	    Args:
	    	corpus (file): file containing the corpus

	    Returns:
	    	sentence (list of tuples): a list of tuples, each of which containing a token and all
	    	the related data (i.e., token, lemma, pos, index, parent, dep)
	    """

		# A corpus can be made of many files
		for corpus_file in corpus_dictionary:
			corpus = corpus_dictionary[corpus_file]

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
