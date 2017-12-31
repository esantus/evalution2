"""Functions to generate annotated corpora from raw text files and to extract data from a corpus.

Example:

    The core of this module is constituted by the three extract functions:
        extract_ngrams(): extract ngrams from a corpus given a set of words.
        extract_patterns(): extract patterns from a corpus given a set of word pairs.
        extract_statistics(): extract word statistics from a corpus given a set of words.

First, we extract a list of words (get_wlist) or pair of words (get_pattern_list())

>>> wlist_fn = join('..', 'data', 'test',  'wordlist_long.csv')
>>> wlist = get_wlist(wlist_fn)

We can use the word list to initialize a Dataset instance (or to coll the add_ functions directly).

>>> dataset = Dataset(wlist)

The dataset class contains three dicionaries which will hold the extracted data.

>>> dataset.ngrams
>>> dataset.patterns
>>> dataset.statistics

We then iterate through the corpus sentence by sentence using get_sentences(filename).

>>> corpus_fn = join('..', 'data', 'test',  'corpora', 'tint_corpus.csv')
>>> for sentence_no, sentence in enumerate(get_sentences(corpus_fn):
...     dataset.add_patterns(sentence, sentence_no)
...     dataset.add_statistics(sentence, sentence_no)

The class also provides a method to save all populated dictionaries in the various csv files.

>>> dataset.save_all('./')

Alternatively, one can call any of the save functions directly by passing a dictionary as the argument.

>>> save_patterns(dataset.patterns, './')

Authors:
    Luca Iacoponi (jacoponi@gmail.com)
    Enrico Santus (esantus@gmail.com)

Todo:
    * Add function to extract from raw text.
    * Add test sets.
    * Use pyannotate and mypy.
"""

import collections
import csv
import gzip
import logging
import math
import os
import pickle
import re
from os.path import join
from typing import *

import flashtext
import tqdm

from evalution import data

#: Corpus fields in an eval corpus
CORPUS_FIELDS = ['token', 'lemma', 'pos', 'index', 'parent', 'dep', 'lemma_i', 'token_i']
logger = logging.getLogger(__name__)


def _cap_type(word: str) -> str:
    """Returns a string describing the capitalization type of word.

    Args:
        word: The word to be analyzed.

    Returns:
        A string indicating the capitalization of word: 'none', 'all', 'first' or 'others'.
    """

    functions = [str.islower, str.isupper, str.istitle]
    for f in functions:
        if f(word):
            return f.__name__[2:]
    return 'other'


def _is_stopword(word: str)-> bool:
    """Returns true if a word is a stopword."""
    if word in data.stopwords or word.endswith("'ll") or word.endswith("'t"):
        return True


def _open_corpus(corpus_path: 'file path', encoding='ISO-8859-2') -> 'IO':
    """Yield an eval corpus reader.

    Args:
        corpus_path: Either a file or a folder. If a folder is passed, the generator yields one corpus at the time.
        encoding: The encoding of the corpus files. Must be the same for all files.

    Yields:
        A corpus reader.
    """

    if os.path.isdir(corpus_path):
        to_open = [(join(corpus_path, f)) for f in os.listdir(corpus_path) if
                   os.path.isfile(join(corpus_path, f))]
    else:
        to_open = [corpus_path]
    if not to_open:
        raise ValueError("'%s' does not contain any corpus " % corpus_path)

    for filename in to_open:
        if os.path.basename(filename).endswith('.gz'):
            corpus = gzip.open(filename, 'rt', encoding=encoding)
        else:
            corpus = open(filename, 'r', encoding=encoding)
        if len(corpus.readline().split("\t")) != len(CORPUS_FIELDS) - 2:
            corpus.close()
            raise ValueError("'%s' is not a valid evalution2 corpus. Try convert_corpus(corpus)." % filename)
        yield corpus


def get_pattern_pairs(wlist: 'file path', separator: 'str' = "\t") -> set:
    """Get a set of unique, symmetric word pairs from a file containing pairs of words.

    Args:
        wlist: Path to the file containing the list of pairs of words to fetch.
        separator: The separator used in the wlist file.

    Returns:
        A set of pairs and their inverse. For example: {('the', 'a'), ('a', 'the'), ('for', 'be'), ('be', 'for')}
    """

    pattern_pairs = set()
    with open(wlist, 'r') as pattern_reader:
        for line in pattern_reader:
            split_line = tuple(word.strip() for word in line.split(separator))
            if len(split_line) == 2:
                if not any(pair in pattern_pairs for pair in (split_line, split_line[::-1])):
                    pattern_pairs.add(split_line)
                    pattern_pairs.add(split_line[::-1])
            else:
                logging.warning("line '%s' in corpus '%s' is not a valid pair" % (line, wlist))
    return pattern_pairs


def get_wlist(wlist_fn: 'file path') -> (set, list):
    """"Generate a set of MWEs and a list of words from a file.

    Args:
        wlist_fn: A file that contains a list of words or MWEs.

    Returns:
        The set of MWEs and a list of words in wlist_fn.
    """
    words = flashtext.KeywordProcessor(case_sensitive=True)
    words.non_word_boundaries.add('-')
    words.non_word_boundaries.add('\'')
    # words.add_keyword_from_file(wlist_fn)
    with open(wlist_fn, 'r', encoding='utf-8') as wlist_reader:
        for line in wlist_reader:
            words.add_keyword(line.strip())
    return words


def get_sentences(corpus_fn: 'file path', file_encoding='utf-8') -> 'eval sentence':
    """
    Yield all the sentences in an eval corpus file as a list of Word namedtuples.

    Args:
        corpus_fn: Filename of the corpus.
        file_encoding: Specify encoding of the corpus

    Yields:
        A list of tuples representing a sentence in the corpus.
    """

    Word = collections.namedtuple('Word', CORPUS_FIELDS)
    line_no = 1
    sentence = []
    lemma_i = 0
    token_i = 0
    possible_eos = False
    # We will exclude everything that does not look like a word, including punctuation.
    word_regex = re.compile("[\w]+[\-']?[\w]+\.?$")
    for corpus_reader in _open_corpus(corpus_fn, encoding=file_encoding):
        with corpus_reader as corpus:
            for line in corpus:
                # The header is read in open_corpus, so we start from line 2.
                line_no += 1
                # Beginning of text or of sentence.
                if line.startswith(('<text', '</text', '<s>')):
                    continue
                elif line.strip() == '</s>':
                    # We store the position of each word, as if the sentence was a string.
                    yield sentence
                    sentence = []
                    lemma_i = token_i = 0
                    possible_eos = False
                else:
                    word_info = line.split() + [lemma_i, token_i]
                    if len(word_info) == len(CORPUS_FIELDS):
                        if not word_regex.match(word_info[1]):
                            continue
                        # Full stop plus a capitalized non-proper name -> It's a missed sentence boundary.
                        if possible_eos and word_info[0].istitle() and word_info[0] not in data.abbreviations:
                            last_word = sentence.pop()
                            last_info = [field for field in last_word]
                            # Remove full stop from lemma and token.
                            last_info[0] = last_info[0][:-1]
                            last_info[1] = last_info[1][:-1]
                            word = Word(*last_info)
                            sentence.append(word)
                            yield sentence
                            sentence = []
                            lemma_i = token_i = 0
                            word_info = line.split() + [lemma_i, token_i]
                            possible_eos = False
                        if word_info[1].endswith('.'):
                            possible_eos = True
                        word = Word(*word_info)
                        sentence.append(word)
                        # Beware, +1 is to include whitespaces.
                        lemma_i += len(word.lemma) + 1
                        token_i += len(word.token) + 1
                    else:
                        logger.warning('Invalid line #%d in %s %s' % (line_no, corpus_fn, line))


def extract_statistics(sentence: 'eval sentence', words: flashtext.KeywordProcessor, statistics: dict) -> dict:
    """Extracts statistical information for each word in words and mwes and stores it in w_stats.

    The dictionary is structured as follows:
    statistics {
        'word': {
            'cap': {lower<str>: freq<int>, upper<str>:freq<int>, title<str>:freq<int>, other<str>:freq<int>
            'freq': freq<int>
            'norm: dict(form<str>: freq<int>): # a dictionary containing normalized forms and their frequencies
            'pos_dep': dict(pos-dep<str>: freq<int>):

    Args:
        sentence: An eval sentence (a list 6-tuples).
        words: KeywordProcessor containing the list of words to count (see _get_wlist()).
        statistics: The dictionary with the statistics to update (or initialize if empty).

    Returns:
        Reference to the updated dictionary.
    """

    all_lemmas = ' '.join([word.lemma for word in sentence])
    matched_indexes = [match[1] for match in words.extract_keywords(all_lemmas, span_info=True)]
    matched_lemmas = [w for w in sentence if w.lemma_i in matched_indexes]
    for word in matched_lemmas:
        lemma = word.lemma
        if lemma not in statistics:
            statistics['last_id'] = statistics.get('last_id', 0) + 1
            statistics[lemma] = {}
            statistics[lemma]["id"] = statistics['last_id']
            statistics[lemma]["freq"] = 0
            statistics[lemma]["cap"] = {cap: 0 for cap in ("lower", "upper", "title", "other")}
            statistics[lemma]["norm"] = collections.Counter()
            statistics[lemma]["pos_dep"] = collections.Counter()
        norm_token = word.token
        # Only the normalized form of proper nouns is left capitalized.
        if not word.token.istitle():
            norm_token = norm_token.lower()
        statistics[lemma]["freq"] += 1
        statistics[lemma]["norm"][norm_token] += 1
        statistics[lemma]["cap"][_cap_type(word.token)] += 1
        statistics[lemma]["pos_dep"][word.pos + "_" + word.dep] += 1
    return statistics


def extract_patterns(sentence: 'eval sentence', word_pairs: set, patterns: dict, save_args: dict = None) -> dict:
    """Returns a dictionary containing all the words between each pair of a set of pair of words.

    The dictionary has the following structure:
    patterns {
            (w1, w1): { # keys are pairs (tuples) of two strings indicating a word (token, or lemma if islemma == True)
            F.dep: ... # the values are another dictionary whose keys are the corpus fields (F.dep, F.lemma, F.token,)
            F.lemma: dict(span<str>: freq<int>) # each dictionary contains the in-between span and its frequency.}}

    Args:
        sentence: A list of 6tuples representing a sentence.
        word_pairs: A set of word pairs and their inverse.
        patterns: The pattern dictionary to be updated or initialized.
        save_args: A dictionary with keys indicating a corpus field. If key is true. save the pattern indicated by it.
            If empty, the following default values are used: {'token':True, 'lemma':True, 'POS':True, 'dep':True}.
    Returns:
        Reference to the updated dictionary.
    """

    if not save_args:
        save_args = {'token': True, 'lemma': True, 'pos': True, 'dep': True}
    else:
        if not any([v for (k, v) in save_args.items() if k in ['token', 'lemma', 'pos', 'dep', 'parent', 'index']]):
            raise ValueError('%s is not a valid dictionary.' % repr(save_args))

    # First time running, initialize dictionary
    if not patterns:
        for pair_id, pair in enumerate(word_pairs):
            patterns[pair] = {arg: collections.Counter() for (arg, val) in save_args.items() if val}
            # Explicitly declare freq = 0 just in case we need to iterate over keys.
            patterns[pair]['freq'] = 0
            patterns[pair]['pair_id'] = pair_id

    all_lemmas = [w.lemma for w in sentence]
    all_tokens = [w.token for w in sentence]
    for pair in word_pairs:
        for i in range(len(all_lemmas)):
            if all_lemmas[i] == pair[0] or all_tokens[i] == pair[0]:
                word1, word2 = pair[0], pair[1]
            else:
                continue
            match_index = i
            for x in range(i, len(all_lemmas)):
                if all_lemmas[x] == word2 or all_tokens[x] == word2:
                    patterns[pair]['freq'] += 1
                    for field, value in save_args.items():
                        if value:
                            all_targets = [getattr(w, field) for w in sentence]
                            in_between = ' '.join(all_targets[match_index + 1:x])
                            patterns[pair][field][in_between] += 1
    return patterns


def extract_ngrams(sentence: 'eval sentence', words, ngrams: dict, win: int = 2,
                   exclude_stopwords: bool=True, istoken: bool=False, pos: bool=True, dep: bool=True) -> dict:
    """Extract_ngrams from a sentence and update an ngram dictionary.

    The ngram dictionary has the following structure:
    ngram {
            'tot_ngram_freq':freq<int>: total number of ngrams
            'tot_word_freq': freq<int>: total number of words
            'word_freq': <dict(word: freq<int>)>: a dict containing all the words (in the ngram format) and their freq.
            'ngram_freq': <dict(ngram<tuple>: dict(freq: freq<int>): the keys of the dictionary are ngram tuples,
                the values are dictionary containing only the key 'freq' and the frequency of the ngram.
                The dictionary can be further populated by add_ngram_probability().}

    Args:
        sentence: An eval sentence.
        words: KeywordProcessor containing the list of words to count (see _get_wlist()).
        ngrams: The ngram dictionary, or an empty dictionary.
        win: The ngram window.
        exclude_stopwords: if True, exclude stopwords from ngrams.
        istoken: If True, match the tokens instead of lemmas.
        pos: If True, ngram includes POS information.
        dep: If true, ngram includes DEP information.

    Returns:
        Reference to the updated dictionary.
    """

    field = 'token' if istoken else 'lemma'
    if not ngrams:
        ngrams.update(dict(tot_word_freq=0, word_freq=collections.Counter(),
                           tot_ngram_freq=0, ngram_freq={}, last_id=0))
    lemmas_to_search = ' '.join([word.lemma for word in sentence[:-win+1]])
    ngrams['tot_word_freq'] += len(sentence)
    matches = collections.deque([match for match in words.extract_keywords(lemmas_to_search, span_info=True)
                                if exclude_stopwords and not _is_stopword(match[0])])
    i = 0
    last_is_stopword = False
    while matches:
        if i == len(sentence):
            if last_is_stopword:
                return ngrams
            # We reached the end, but there are still items in the deque.
            # logging.warning("Missing index for %s" % repr(matches[0]))
            matches.popleft()
            i = 0
            continue
        _, match_begin, match_end = matches[0]
        word = sentence[i]
        if word.lemma_i == match_begin:
            for j in range(i+1, len(sentence)):
                word = sentence[j]
                if word.lemma_i == match_end + 1:
                    if not last_is_stopword:
                        after_target_i = j
                    if exclude_stopwords:
                        if word.token in data.stopwords:
                            # Push match end to the end of the stopword
                            match_end += len(word.lemma) + 1
                            last_is_stopword = True
                            continue
                    ngram_end_index = j + (win - 1)
                    if ngram_end_index > len(sentence):
                        break
                    ngram_first_slice = sentence[i:after_target_i]
                    ngram_second_slice = sentence[j:ngram_end_index]

                    if pos and not dep:
                        ngram = tuple(' '.join((str(getattr(w, field)) + '-' + w.pos
                                                for w in ngram_first_slice)))
                        ngram += tuple(str(getattr(w, field)) + '-' + w.pos
                                       for w in ngram_second_slice if w.lemma not in data.stopwords)
                    elif pos and dep:
                        ngram = (' '.join(tuple(str(w.dep) + ':' + str(getattr(w, field)) + '-' + w.pos
                                 for w in ngram_first_slice)), )
                        ngram += tuple((str(w.dep + ':' + str(getattr(w, field)) + '-' + w.pos)
                                       for w in ngram_second_slice if w.lemma not in data.stopwords))
                    else:
                        ngram = tuple(' '.join(getattr(w, field) for w in ngram_first_slice))
                        ngram += tuple(getattr(w, field) for w in ngram_second_slice
                                       if w.lemma not in data.stopwords)
                    ngram_lemmas = ' '.join([w.lemma for w in sentence[i:ngram_end_index]
                                             if w.lemma not in data.stopwords])

                    if ngram not in ngrams['ngram_freq']:
                        ngrams['last_id'] += 1
                        ngrams['ngram_freq'][ngram] = {'freq': 0, 'ngram_id': ngrams['last_id'] - 1}
                    ngrams['ngram_freq'][ngram]['freq'] += 1
                    ngrams['ngram_freq'][ngram]['lemmas'] = ngram_lemmas
                    ngrams['tot_ngram_freq'] += 1
                    for item in ngram:
                        ngrams['word_freq'][item] += 1
                    matches.popleft()
                    last_is_stopword = False
                    i = after_target_i - 1
                    break
        i += 1
    return ngrams


def add_ngram_probability(ngrams: dict, plmi: bool = False) -> dict:
    """Add a probability value to each ngram as ngrams[ngram_freq][ngram]['probability'].

    Args:
        ngrams: The ngram dictionary to update.
        plmi: Use plmi.

    Returns
        The updated ngram dictionary containing a probability field for each ngram with freq. > 3.
    """

    for ngram in ngrams['ngram_freq']:
        curr_ngram = ngrams['ngram_freq'][ngram]
        # In calculating ppmi, put a cutoff of freq > 3 to avoid rare events to affect the rank
        if curr_ngram['freq'] < 4:
            probability = 0
        else:
            ngram_prob = float(curr_ngram['freq']) / ngrams['tot_ngram_freq']
            # Initializing the variable to calculate the probability of components as independent events
            components_prob = 1
            for word in ngram:
                components_prob *= float(ngrams['word_freq'][word]) / ngrams['tot_word_freq']
            probability = math.log(ngram_prob / components_prob)  # ppmi
            if plmi:
                probability *= curr_ngram['freq']  # plmi
        ngrams['ngram_freq'][ngram]['probability'] = probability
    return ngrams


def save_ngrams(ngrams: dict, outfile_path: 'file path'):
    """Save ngrams to a tsv file.

    Args:
        ngrams: The ngram dictionary.
        outfile_path: The filename of the output file.

    Returns:
        True if file is successfully written.
    """
    # save probability only if at least one element has probability > 0
    probability = any([ngram['probability'] for _, ngram in ngrams['ngram_freq'].items() if 'probability' in ngram])
    with open(outfile_path, 'w', encoding='utf-8', newline='') as outfile:
        ngram_writer = csv.writer(outfile)
        header = ['ngram_id', 'ngram', 'lemmas', 'freq']
        if probability:
            header.append('probability')
        ngram_writer.writerow(header)
        for ngram, ngram_d in ngrams['ngram_freq'].items():
            row = [ngram_d['ngram_id'], ' '.join(ngram), ngram_d['lemmas'], ngram_d['freq']]
            if probability:
                row.append(ngram_d['probability'])
            ngram_writer.writerow(row)
    logging.info('%s saved.' % outfile_path)


def save_ngram_stats(ngrams: dict, statistics: dict, outfile_path: 'file path'):
    """Save a file mapping ngrams_id to word_ids.

    Args:
        ngrams: The ngram dictionary.
        statistics: The statistics dictionary.
        outfile_path: The filename of the output file.

    Returns:
        True if file is successfully written.
    """

    with open(outfile_path, 'w', encoding='utf-8', newline='') as outfile:
        ngram_writer = csv.writer(outfile)
        header = ['ngram_id', 'word_id', 'ngram_index']
        ngram_writer.writerow(header)
        for _, ngram_d in ngrams['ngram_freq'].items():
            for ngram_index, lemma in enumerate(ngram_d['lemmas']):
                if lemma in statistics:
                    word_id = statistics[lemma]['id']
                    row = [ngram_d['ngram_id'], word_id, ngram_index]
                    ngram_writer.writerow(row)
    logging.info('%s saved.' % outfile_path)


def save_patterns(patterns: dict, outfile_path: 'file path'):
    """Save patterns dictionary in a csv file.

    Args:
        patterns: The patterns dictionary.
        outfile_path: The filename of the output file.

    Returns:
        True if file is successfully written.
    """

    with open(outfile_path, 'w', encoding='utf-8', newline='') as outfile:
        pattern_writer = csv.writer(outfile)
        header = ['pattern_id', 'pair_id', 'word1', 'word2', 'freq.', 'type', 'context', 'frequency']
        pattern_writer.writerow(header)
        col_id = 0
        for pair, n_types in patterns.items():
            pair_freq = n_types['freq']
            pair_id = n_types['pair_id']
            for n_type, contexts in n_types.items():
                if isinstance(contexts, collections.Counter):
                    for context, frequency in contexts.items():
                        row = [col_id, pair_id, pair[0], pair[1], pair_freq, n_type, context, frequency]
                        pattern_writer.writerow(row)
                        col_id += 1
    logging.info('%s saved.' % outfile_path)


def save_statistics(statistics: dict, outfile_path: 'file path'):
    """Save statistics dictionary in a csv file.

    Args:
        statistics: The statistics dictionary.
        outfile_path: The filename of the output file.

    Returns:
        True if file is successfully written.
    """

    filenames = ["{}_{}.csv".format(os.path.splitext(outfile_path)[0], suffix)
                 for suffix in ['words', 'forms', 'posdep']]
    open_args = dict(mode='w', encoding='utf-8', newline='')
    with open(filenames[0], **open_args) as outfile_words, open(
              filenames[1], **open_args) as outfile_norm, open(
              filenames[2], **open_args) as outfile_posdep:

        word_f = csv.writer(outfile_words)
        norm_f = csv.writer(outfile_norm)
        posdep_f = csv.writer(outfile_posdep)
        header_main = ['stat_id', 'word', 'freq.', 'cap_lower', 'cap_upper', 'cap_title', 'cap_other']
        header_norm = ['norm_id', 'stat_id', 'norm.', 'freq.']
        header_posdep = ['posdep_id', 'stat_id', 'posdep', 'freq.']
        word_f.writerow(header_main)
        norm_f.writerow(header_norm)
        posdep_f.writerow(header_posdep)
        word_id, norm_id, posdep_id = (0 for _ in range(3))

        for word, attribute in statistics.items():
            if word == 'last_id':
                continue
            cap = attribute['cap']
            row = [word_id, word, attribute['freq'], cap['lower'], cap['upper'], cap['title'], cap['other']]
            word_f.writerow(row)
            for attr_name, attr_values in attribute.items():
                if attr_name == 'norm':
                    for norm, freq in attr_values.items():
                        row = [norm_id, word_id, norm, freq]
                        norm_f.writerow(row)
                        norm_id += 1
                elif attr_name == 'pos_dep':
                    for posdep, freq in attr_values.items():
                        row = [posdep_id, word_id, posdep, freq]
                        posdep_f.writerow(row)
                        posdep_id += 1
            word_id += 1
    for filename in filenames:
        logging.info('%s saved.' % filename)


class Dataset:
    def __init__(self, word_list=None, ngram_list=None, pattern_list=None, pickle_every=None,
                 pickle_out=os.getcwd(), overwrite_pickles=False):
        """Dataset object containing ngrams, statistics and pattern dictionaries, and methods to extract and save them.

        Args:
            word_list: set of words to search the statistics for.
            ngram_list: set of words to use to extract the ngrams.
            pattern_list: set of word pairs to use to extract the patterns.
            pickle_every: Dump pickle file for ngrams, patterns and stats after the indicated no. of sentences.
            pickle_out: Path to folder that stores the pickled files.
            overwrite_pickles: if set to False, raise a warning when trying to write on a folder with existing pickles.
        """

        self._pickle_names = ('ngrams.p', 'patterns.p', 'statistics.p')
        self._overwrite_pickles = None
        self.start_from = 0
        self.pickled = None
        self.ngram_list = ngram_list
        self.word_list = word_list
        self.pattern_list = pattern_list
        self.pickle_out = pickle_out
        self.pickle_every = pickle_every
        self.ngrams, self.patterns, self.statistics = (dict() for _ in range(3))
        self.overwrite_pickles = overwrite_pickles

    @property
    def overwrite_pickles(self):
        return self._overwrite_pickles

    @overwrite_pickles.setter
    def overwrite_pickles(self, overwrite):
        if not overwrite and any(os.path.exists(join(self.pickle_out, pickle_file))
                                 for pickle_file in self._pickle_names):
            logging.warning('Pickle files exist in %s. Set overwrite_pickle=True if you want to overwrite them. '
                            'Pickles will not be dumped.' % self.pickle_out)
            self._overwrite_pickles = False
        else:
            self._overwrite_pickles = True

    class Pickler:
        def __init__(self, to_pickle):
            self.to_pickle = to_pickle
            self.pickle_file = to_pickle + '.p'

        def __call__(self, add_func):
            def pickled_add(instance, sentence, sentence_no=None):
                if sentence_no and sentence_no < instance.start_from:
                    return
                add_func(instance, sentence)
                if instance.pickle_every and instance.overwrite_pickles and not (sentence_no+1) % instance.pickle_every:
                    pickle.dump(getattr(instance, self.to_pickle),
                                open(join(instance.pickle_out, self.pickle_file), 'wb'))
                    logging.info('%s pickled in: %s' % (self.pickle_file, instance.pickle_out))
            return pickled_add

    @Pickler(to_pickle='ngrams')
    def add_ngrams(self, sentence):
        self.ngrams = extract_ngrams(sentence, self.ngram_list, self.ngrams)

    @Pickler(to_pickle='patterns')
    def add_patterns(self, sentence):
        self.patterns = extract_patterns(sentence, self.pattern_list, self.patterns)

    @Pickler(to_pickle='statistics')
    def add_statistics(self, sentence):
        self.statistics = extract_statistics(sentence, self.word_list, self.statistics)

    def add_ngram_prob(self):
        self.ngrams = add_ngram_probability(self.ngrams)

    def load_pickles(self, pickle_names):
        # Assuming it is a folder
        if isinstance(pickle_names, str):
            pickles = [join(pickle_names, filename) for filename in self._pickle_names]
            if not all(os.path.exists(file) for file in pickles):
                raise ValueError('Missing pickles in "%s". load_pickles() requires %s'
                                 % (pickle_names, ', '.join(self._pickle_names)))
            if not os.path.exists(join(pickle_names, 'last_sentence_index.p')):
                self.start_from = 0
            else:
                start_from = pickle.load(open(join(pickle_names, 'last_sentence_index.p'), 'rb'))
                if not isinstance(start_from, int):
                    raise TypeError('last_sentence_index.p must be of type int.')
                self.start_from = start_from
                logging.info('Found last_index.p. Resuming from sentence number ' + str(self.start_from))
        elif isinstance(pickle_names, collections.abc.Sequence) and len(pickle_names) == 4:
            pickles = pickle_names[:3]
            self.start_from = pickles[3] if pickles[3] else 0
        else:
            raise TypeError('pickle_names must be a string or a sequence of len 4.')

        self.ngrams, self.patterns, self.statistics = (pickle.load(open(pickle_file, 'rb')) for pickle_file in pickles)
        logging.info('Pickles loaded.')

    def save_all(self, output_dir=os.getcwd()):
        if self.ngrams:
            save_ngrams(self.ngrams, join(output_dir, 'ngrams.csv'))
            if self.statistics:
                save_ngram_stats(self.ngrams, self.statistics, join(output_dir, 'ngram_words.csv'))
        if self.patterns:
            save_patterns(self.patterns, join(output_dir, 'patterns.csv'))
        if self.statistics:
            save_statistics(self.statistics, join(output_dir, 'statistics.csv'))


def test_data():
    """Save ngrams, patterns and statistics to a file using test data."""
    data_dir = os.path.normpath(join(os.path.dirname(__file__), os.pardir + '/data'))
    test_dir = join(data_dir, 'test')
    output_dir = join(data_dir, 'output')
    pickle_dir = join(output_dir, 'pickle')

    wlist_fn = join(test_dir, 'wordlist_long.csv')
    plist_fn = join(test_dir, 'patterns.csv')
    corpus_fn = join(test_dir, 'tiny_corpus.csv')

    nlist_fn = wlist_fn
    dataset = Dataset(get_wlist(wlist_fn), get_wlist(nlist_fn), get_pattern_pairs(plist_fn), 5000, pickle_dir, True)
    dataset.load_pickles(pickle_dir)
    corpus_len = sum(1 for _ in get_sentences(corpus_fn))
    for sentence_no, sentence in enumerate(tqdm.tqdm(get_sentences(corpus_fn), mininterval=0.5, total=corpus_len)):
        dataset.add_ngrams(sentence, sentence_no)
        dataset.add_patterns(sentence, sentence_no)
        dataset.add_statistics(sentence, sentence_no)
    dataset.add_ngram_prob()
    dataset.save_all(output_dir)


def main():
    test_data()

if __name__ == "__main__":
    # To run as a script use python -m evalution.corpus from parent folder.
    main()
