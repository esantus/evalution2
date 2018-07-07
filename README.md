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

*allwordsenses* - maps word id to their synsets

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

A small API is provided that processes the most frequent queries.

`all_words(lang)`: return a list of words from lang

`all_rels(rel)`: return a list of all words related by rel.
    If rel is None, returns a list of all words related by any rel.

`which_rels(w1, w2)`: return the list of relations between w1 and w2

`are_rel(w1, w2, rel)`: return True if w1 and w2 are related by a rel,
    if rel is None, return True if w1 and w2 are related by any rel. 
    Return False otherwise.

