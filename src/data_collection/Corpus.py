# -*- coding: <coding> -*-

import os
import sys
import re
import gzip
import pickle
import numpy as np

import codecs
import traceback

import nltk
from nltk import word_tokenize
from nltk.stem import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

import panda as pd

from docopt import docopt
from collections import defaultdict
from composes.utils import io_utils
from composes.semantic_space.space import Space
from scipy.sparse import coo_matrix, csr_matrix
from composes.matrix.sparse_matrix import SparseMatrix



class Corpus:
	"""
	This class extracts metadata from corpora.

	Args:
    	path: path of the corpora
	"""
	
	def __init__(self, corpus_path, wordlist_path):

		"""
		Initializes the main attributes.

			:param corpus_path: path of the corpus directory
			:param wordlist_path: path of the wordlist
			:type corpus_path: string
			:type wordlist_path: string
		"""
		self.corpus_path = corpus_path
		self.wordlist_path = wordlist_path
		self.wl = []

		self.CASE_SENSITIVE = 0
		self.field = {"token":0, "lemma":1, "stem":2, "pos":3, "index":4, "parent":5, "dep":6}
		self.stopwords = set(stopwords.words('english'))

		self.data = {}
		self.load_data()

		self.wordpos = nltk.ConditionalFreqDist((w.lower(), t) for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))



	def findWord(self, w, flags=re.IGNORECASE):
		"""
		Returns all the instances of a word in a sentence

			:return instances of words in a sentence
			:rtype list of found words
		"""

		w= w.replace(".", "\\.")
		w= w.replace("*", "\\*")
		W= w.replace("?", "\\?")
		return re.compile(r'\b({0})\b'.format(w), flags).findall



	def whichCapit(self, word):
		"""
		Returns the capitalization type
			
			:return capitalization type
			:rtype string which can be used as a key for the capitalization dictionary
		"""

		option = "none" if not any(map(str.isupper, str(word))) else ""

		if option == "":
			option = "all" if not any(map(str.islower, str(word))) else ""

		if option == "":
			option = "first" if (str.isupper(str(word[0])) and not any(map(str.isupper, str(word[1:])))) else "others"

		return option
		


	def load_data(self):
		"""
		Uploads the wordlist and create a dictionary with every word as key
		"""

		try:
			with open(self.wordlist_path, "r") as wl_file:
				for word in wl_file:
					word = word.strip()

					data = {
						'freq': 0,
						'norm': {}, # All inflections after token.lower()
						'not_norm': {"first": 0, "all": 0, "none": 0, "others": 0},
						'pos_dep': {}, # POS-dep distribution for lemma (this allows to derive POS and Deps)
						'collocations': []
					}

					if word.isalnum():
						self.wl.append(unicode(word))
						self.data[unicode(word)] = data

			self.wl = list(set(self.wl))
			print("Wordlist has been loaded")
		except Exception as error:
			print("It was not possible to open: " + self.wordlist_path)
			print("Error:", error)



	def print_data(self):
		for word in self.data:
			print([word, self.data[word]["freq"], self.data[word]["norm"], self.data[word]["not_norm"], self.data[word]["pos_dep"]])



	def get_statistics_from_parsed_corpus(self):
		"""
		Returns a dataframe containing corpus-based information about words that are found in a wordlist

			:return corpus-based informationa about words in wordlist
			:rtype panda dataframe
		"""

		try:
			f_out_analiz = codecs.open("analized_corpus.txt", "w", "utf-8")
			f_out_token = codecs.open("tokenized_corpus.txt", "w", "utf-8")
			f_out_lemma = codecs.open("lemmatized_corpus.txt", "w", "utf-8")
		except Exception as e:
			print("It is not possible to open the file for writing: ", e)

		try:
			corpus_files = sorted([self.corpus_path + '/' + file for file in os.listdir(self.corpus_path) if file.endswith('.gz')])

			for file_num, corpus_file in enumerate(corpus_files):
				print("Processing corpus file %s (%d/%d)..." % (corpus_file, file_num+1, len(corpus_files)))

				with codecs.open("statistics.txt", "w", "utf-8") as f_statistics:
					self.save_data(f_statistics)

				text = ""
				i = 0

				for sentence in self.get_sentences_from_parsed_corpus(corpus_file):

					# Write the linear corpus with "DEP-Lemma-POS" tokens.
					f_out_token.write(sentence[0][0] + "\n")
					f_out_lemma.write(sentence[0][1] + "\n")
					f_out_analiz.write(sentence[0][2] + "\n")

					match = ""
					word = ""

					# Find it among the tokenized words
					for match_index in [i for (i, [w, l, s, p, useless, parent, d]) in enumerate(sentence[1:])]:

						# If we have 7 fields for the given match
						if match_index <= len(sentence) and len(sentence[match_index])==7:
							match = sentence[match_index][self.field["token"]]
							word = sentence[match_index][self.field["lemma"]]

							if word in self.data:

								# Save lemma frequency
								self.data[word]["freq"] += 1

								# Save frequency of inflected forms
								if match.lower() not in self.data[word]["norm"]:
									self.data[word]["norm"][match.lower()] = 0
								self.data[word]["norm"][match.lower()] += 1

								# Save capitalization type
								self.data[word]["not_norm"][self.whichCapit(match)] += 1

								# Save POS-Dep
								if sentence[match_index][self.field["pos"]]+"-"+sentence[match_index][self.field["dep"]] not in self.data[word]["pos_dep"]:
									self.data[word]["pos_dep"][sentence[match_index][self.field["pos"]]+"-"+sentence[match_index][self.field["dep"]]] = 0
								self.data[word]["pos_dep"][sentence[match_index][self.field["pos"]]+"-"+sentence[match_index][self.field["dep"]]] += 1

						#else:
							# If the word does not have seven fields or it is over the sentence end
						#	print(sentence, "if match_index <= len(sentence) and self.field[\"token\"] <= len(sentence[match_index]) and self.field[\"lemma\"] <= len(sentence[match_index])")

						#Collocations
							#Bigrams, Threegrams, Fourgrams
								#Dep:TargetToken-POS Dep:ContextToken-POS
								#Dep:TargetLemma-POS Dep:ContextLemma-POS

		except Exception as error:
			print("It was not possible to open the corpora in: " + self.corpus_path)
			print("Error:", error)
			tb = sys.exc_info()[-1]
    		print(traceback.extract_tb(tb, limit=1)[-1][1])


		return



	def load_statistics(self, filename):
		"""
		Returns a list of [lemma, freq, inflect_dictionary, normaliz_dictionary, pos_dep_dictionary]
		"""

		# Fields of the datastructure
		lemma_string = 0
		tot_freq_int = 1
		inflect_dict = 2
		capit_dict = 3
		pos_dep_dict = 4

		with open(filename, "r") as f1:

				for line in f1:
					item1 = line.strip().split("\t")

					item1[tot_freq_int] = int(item1[tot_freq_int])
					item1[inflect_dict] = eval(item1[inflect_dict])
					item1[capit_dict] = eval(item1[capit_dict])
					item1[pos_dep_dict] = eval(item1[pos_dep_dict])

					first.append(item1)

		return first



	def not_in_statistics(self, wordlist, data):
		"""
		Returns a subset of the wordlist containing only terms that are not saved in the data
		"""

		not_in = []

		for word in wordlist:
			if word not in map(itemgetter(0), data):
				not_in.append(word)

		return not_in


	def merge_and_print_merged(self, file1, file2):
		"""
		Returns a list of dataframes merging two metadata files having the following fields:
			- lemma (string)
			- freq (int)
			- inflections (dictionary: freq)
			- normalization (dictionary: freq)
			- pos-dep (dictionary: freq)
		"""

		# Fields of the datastructure
		lemma_string = 0
		tot_freq_int = 1
		inflect_dict = 2
		capit_dict = 3
		pos_dep_dict = 4

		# Loading the lists
		first = self.load_statistics(file1)
		second = self.load_statistics(file2)

		# Preparing the list of results
		result = []

		# To be used later to check that all the values of second are moved into results
		used = [0] * len(second)

		# For every line of first, do the mapping with the correspective of second
		for item1 in first:
			
			try:
				index = zip(*second)[0].index(item1[0])
			except:
				try:
					index = map(itemgetter(0), second).index(item1[0])
				except:
					print "\n\n\n%s NOT FOUND\n\n\n" % item1[0]
					raw_input()
					index = -1

			if index != -1:
				used[index] = 1

				if item1[0] == second[index][0]:
					item2 = second[index]

					item1[tot_freq_int] += item2[tot_freq_int]

					for key in item2[inflect_dict]:
						if key in item1[inflect_dict]:
							item1[inflect_dict][key] += item2[inflect_dict][key]
						else:
							item1[inflect_dict][key] = item2[inflect_dict][key]

					for key in item1[capit_dict]:
						item1[capit_dict][key] += item2[capit_dict][key]

		                        for key in item2[pos_dep_dict]:
		                                if key in item1[pos_dep_dict]:
		                                        item1[pos_dep_dict][key] += item2[pos_dep_dict][key]
		                                else:
		                                        item1[pos_dep_dict][key] = item2[pos_dep_dict][key]

					result.append(item1)


		# For every line of second that was not mapped into an existing one from first
		for i, value in enumerate(used):
			# If the value was not changed when mapping into 1
			if value == 0:
				if second[i][0] not in map(itemgetter(0), result):
					print "Adding ", second[i]
					raw_input("Click something")
					result.append(second[i])
				else:
					print "We could not add ", second[i] 
		                        raw_input("Click something")


		return result



	def print_data(self):
		for word in self.data:
			print([word, self.data[word]["freq"], self.data[word]["norm"], self.data[word]["not_norm"], self.data[word]["pos_dep"]])



	def save_data(self, f_out):
		for word in self.data:
			f_out.write(str(word) + "\t" + str(self.data[word]["freq"]) + "\t" + str(self.data[word]["norm"]) + "\t" + str(self.data[word]["not_norm"]) + "\t" + str(self.data[word]["pos_dep"]) + "\n")




	def get_sentences_from_parsed_corpus(self, corpus_file):
	    """
	    Returns all the (content) sentences in a corpus file

		    :param corpus_file: the corpus file name
		    :type corpus_file: string
		    :return: the next sentence (yield)
		    :rtype: list of tuples
	    """

	    with gzip.open(corpus_file, 'r') as f_in:

	        s = []
	        full_sentence_token = full_sentence_lemma = full_sentence_analized = ""
	        porter_stemmer = PorterStemmer()

	        for line in f_in:
	            line = line.decode('ISO-8859-2')

	            # Ignore start and end of doc
	            if '<text' in line or '</text' in line or '<s>' in line:
	                continue
	            # End of sentence
	            elif '</s>' in line:
	            	s.insert(0, (full_sentence_token.strip(), full_sentence_lemma.strip(), full_sentence_analized.strip()))
	                yield s
	                s = []
	                full_sentence_token = full_sentence_lemma = full_sentence_analized = ""
	            else:
	                try:
	                    word, lemma, pos, index, parent, dep = line.split()
	                    stem = porter_stemmer.stem(word)
	                    s.append((word, lemma, stem, pos, int(index), int(parent), dep))

	                    if dep not in ["P", "DEP"] and pos not in ["SENT", ")", "(", ",", "[", "]", "{", "}", "#", "###", "##", "|", "*"] and lemma not in ["@card@", "..", "..."]: # and lemma not in self.stopwords:
	                    	full_sentence_analized = full_sentence_analized + " " + dep+"_"+lemma+"_"+pos
	                    	full_sentence_token = full_sentence_token + " " + word
	                    	full_sentence_lemma = full_sentence_lemma + " " + lemma

	                except:
	                    continue



x = Corpus("/Users/esantus/Projects/temp_corp", "/Users/esantus/Projects/temp_corp/wordlist.txt")
x.get_statistics_from_parsed_corpus()
raw_input("PRINTING EVERYTHING")
x.print_data()


















"""
	def get_statistics_from_wordlist(self):
		"""#Return a dataframe containing corpus-based information about words in wordlist

			#:return corpus-based informationa about words in wordlist
			#:rtype dataframe
		"""

		try:
			f_out_analiz = codecs.open("analized_corpus.txt", "w", "utf-8")
			f_out_token = codecs.open("tokenized_corpus.txt", "w", "utf-8")
			f_out_lemma = codecs.open("lemmatized_corpus.txt", "w", "utf-8")
		except Exception as e:
			print("It is not possible to open the file for writing: ", e)


		try:
			corpus_files = sorted([self.corpus_path + '/' + file for file in os.listdir(self.corpus_path) if file.endswith('.gz')])

			for file_num, corpus_file in enumerate(corpus_files):
				print("Processing corpus file %s (%d/%d)..." % (corpus_file, file_num+1, len(corpus_files)))

				text = ""
				#i = 0

				for sentence in self.get_sentences_from_parsed_corpus(corpus_file):
					print(sentence)

					#if i >= 200:
					#	break

					#i += 1

					# Write the linear corpus with "DEP-Lemma-POS" tokens.
					f_out_token.write(sentence[0][0] + "\n")
					f_out_lemma.write(sentence[0][1] + "\n")
					f_out_analiz.write(sentence[0][2] + "\n")


					# For every word in wordlist (including MWE):
					for word in self.wl:

						found = False

						# Find it among the tokenized words
						for match_index in [i for w, l, s, p, i, parent, d in sentence[1:]] if str(l.lower()) == word.lower()]:
							match = sentence[match_index][self.field["token"]]

							# Save lemma frequency
							self.data[word]["freq"] += 1

							# Save frequency of inflected forms
							if match.lower() not in self.data[word]["norm"]:
								self.data[word]["norm"][match.lower()] = 0
							self.data[word]["norm"][match.lower()] += 1

							# Save capitalization type
							self.data[word]["not_norm"][self.whichCapit(match)] += 1

							# Save POS-Dep
							if sentence[match_index][self.field["pos"]]+"-"+sentence[match_index][self.field["dep"]] not in self.data[word]["pos_dep"]:
								self.data[word]["pos_dep"][sentence[match_index][self.field["pos"]]+"-"+sentence[match_index][self.field["dep"]]] = 0
							self.data[word]["pos_dep"][sentence[match_index][self.field["pos"]]+"-"+sentence[match_index][self.field["dep"]]] += 1

							# Make sure the string matching process is not executed
							found = True

						# If it is not found among the lemmas, check (case-insensitive) in the tokenized sentence
						if found == False and self.findWord(word)(sentence[0][0]): # Found in case insensitive

							for match in self.findWord(word)(sentence[0][0]):
								# Save lemma frequency
								self.data[word]["freq"] += 1

								# Save frequency of inflected forms
								if match.lower() not in self.data[word]["norm"]:
									self.data[word]["norm"][match.lower()] = 0
								self.data[word]["norm"][match.lower()] += 1

								# Save capitalization type
								self.data[word]["not_norm"][self.whichCapit(match)] += 1

								# Save POS-Dep
								for pos in list(self.wordpos[word]):
									if pos not in self.data[word]["pos_dep"]:
										self.data[word]["pos_dep"][pos] = 0
									self.data[word]["pos_dep"][pos] += 1

						#Collocations
							#Bigrams, Threegrams, Fourgrams
								#Dep:TargetToken-POS Dep:ContextToken-POS
								#Dep:TargetLemma-POS Dep:ContextLemma-POS

		except Exception as error:
			print("It was not possible to open the corpora in: " + self.corpus_path)
			print("Error:", error)

		return
"""












