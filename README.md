# Evalution

Evalution is a collection of tools for the EVALuation
models of semantic relations.

It consists of four components: 

* data extraction library:
    * A library to extract ngrams, patterns 
    and frequencies from corpora.
* datasets (data + api):
    * A dataset of semantic relations.
* baseline:
    * A baseline model.
* evaluation library:
    * A library for the automatic evaluation of semantic models.

## Data extraction library

#### raw_data (data/)
*test/corpora:* Contains the corpora you want to extract the data from.
They must be  in csv format with the following header: WORD, LEMMA, POS, INDEX, PARENT, DEP.

*test/wordlist:* the list of words used to extract frequencies and ngrams.

*test/patterns:* a list of pairs used to extract patterns (string between the two target words).

#### corpus.py

This module contains a set of functions used to extract data from
the corpora. The functions are optimized to extract a large number of words simultaneously.

The main functions, classes and methods are:

`_open_corpus(fpath)` preprocess (sanity check and concatenation)
 and yield a corpus as a file object.

 `Dataset(object)` This object holds all the info about your data.
It need a list of words or word pairs to be processed, and then it stores
useful information about it.

When processing the files, the class allows us to pickle its state, so if the process is interrupted it can be resumed.

The three crucial attributes are:

* `ngrams` an NgramCollection object (see below).
* `frequencies` a dictionary of word frequencies, where k is a string representing a word, and v is a WordFrequency object.
* `patterns` a dictionary of pattern frequencies, where k is a tuple of two words, and v is  a PatternFrequency object.

TBF

## dataset

The dataset is an SQLite database (also available in MySQL format) 
that contains information about the semantic relations. Some useful tables are described below.

*allwordsenses* - maps word id to their synsets. Words with the same sense are
synonyms. Words with the same word_id are homographs. 

* word_id
* language_id
* wordsense_id

*synsetrelation* - maps a pair of synsets to a relation

 * sourcesynset_id
 * relation_id
 * targetsynset_id
 
*word* - list of all words
 * word_id
    * bank -> 347049
 * word_value
 
*language* - list of languages

* language_id
    * en -> 6
* language_value

*relationname* - list of relations 

* relationname_id
* relationname_value

*sysnetdomain* - list of domains
* synsetdomain_id
* synsetdomain_value (e.g. agricolture, advertising, etc.)

*Domain2synset* - how likely is a synset to be in a domain

* synset_id
* domain_id
* score

#### dataset API


