"""Functions to generate annotated corpora from raw text files, and to create support tables for the gold dataset.

TODO:
    * Use flash-text if keywords > 1000.
    * Add test sets.
    * Use pyannotate and mypy.
    * Add extract function.
"""

import collections
import csv
import gzip
import logging
import math
import os
import pickle
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


def _get_pattern_pairs(wlist: 'file path', separator: 'str' = "\t") -> set:
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


def _get_wlist(wlist_fn: 'file path') -> (set, list):
    """"Generate a set of MWEs and a list of words from a file.

    Args:
        wlist_fn: A file that contains a list of words or MWEs.

    Returns:
        The set of MWEs and a list of words in wlist_fn.
    """
    words = flashtext.KeywordProcessor()
    # words.add_keyword_from_file(wlist_fn)
    with open(wlist_fn, 'r', encoding='utf-8') as wlist_reader:
        for line in wlist_reader:
            words.add_keyword(line.strip())
    return words


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
            raise ValueError("'%s' is not a valid evalution2 corpus. Use the function "
                             "convert_corpus(corpus) to create an evalution2 corpus" % filename)
        yield corpus


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
                else:
                    word_info = line.split() + [lemma_i, token_i]
                    if len(word_info) == len(CORPUS_FIELDS):
                        word = Word(*word_info)
                        sentence.append(word)
                        # Beware! +1 is to artificially include whitespaces.
                        lemma_i += len(word.lemma) + 1
                        token_i += len(word.token) + 1
                    else:
                        logger.warning('Invalid line #%d in %s %s' % (line_no, corpus_fn, line))


def extract_statistics(sentence: 'eval sentence', words: flashtext.KeywordProcessor, statistics: dict) -> bool:
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
        True if dictionary is successfully updated/created.
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
    return True


def extract_patterns(sentence: 'eval sentence', word_pairs: set, patterns: dict, save_args: dict = None) -> bool:
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
        True if dictionary is successfully updated/created.
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
        for i in range(0, len(all_lemmas)):
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
    return True


def extract_ngrams(sentence: 'eval sentence', words, ngrams: dict, mwes: set = None, win: int = 2,
                   exclude_stopwords: bool=True, istoken: bool=False, pos: bool=True, dep: bool=True) -> bool:
    """Extract_ngrams from a sentence and update an ngram dictionary.

    The ngram dictionary has the following structure:
    ngram {
            'tot_ngram_freq':freq<int>: total number of ngrams
            'tot_word_freq': freq<int>: total number of words
            'word_freq': <dict(word: freq<int>)>: a dict containing all the words (in the ngram format) and their freq.
            'ngram_freq': <dict(ngram<tuple>: dict(freq: freq<int>): the key of the dictionary are ngram pairs (tuples),
                the values are dictionary containing only the key 'freq' and the frequency of the ngram.
                The dictionary can be further populated by add_ngram_probability().}

    Args:
        sentence: An eval sentence.
        words: KeywordProcessor containing the list of words to count (see _get_wlist()).
        ngrams: The ngram dictionary, or an empty dictionary.
        mwes: The set of mwes to extract.
        win: The ngram window.
        exclude_stopwords: if True, exclude stopwords from ngrams.
        istoken: If True, match the tokens instead of lemmas.
        pos: If True, ngram includes POS information.
        dep: If true, ngram includes DEP information.
    Returns:
        True if dictionary is successfully updated/created.
    """

    if not ngrams:
        ngrams.update(dict(tot_word_freq=0, word_freq=collections.Counter(),
                           tot_ngram_freq=0, ngram_freq={}, last_id=0))
    if exclude_stopwords:
        stopwords = data.stopwords
        sentence = [w for w in sentence if w.token not in stopwords]

    all_lemmas = ' '.join([word.lemma for word in sentence])
    matched_indexes = [match[1] for match in words.extract_keywords(all_lemmas, span_info=True)]


    field = 'token' if istoken else 'lemma'
    ngrams['tot_word_freq'] += len(sentence)
    if pos and not dep:
        rich_sentence = [str(getattr(w, field)) + '-' + w.pos for w in sentence]
    elif pos and dep:
        rich_sentence = [str(w.dep + ':' + str(getattr(w, field)) + '-' + w.pos) for w in sentence]
    else:
        rich_sentence = [getattr(w, field) for w in sentence]

    # Generates the ngrams
    for index in matched_indexes:

        end_window_index = second_ngram_index + win - 1
        if end_window_index > len(sentence):
            break
        if word in wordlist:
            context_slice = slice(second_ngram_index, end_window_index)
            context = ' '.join(rich_sentence[context_slice])
            ngram = (raw_word, context)
            all_lemma = [w.lemma for w in sentence]
            context = ' '.join(all_lemma[context_slice])
            ngram_lemmas = (word, context)
            if ngram not in ngrams['ngram_freq']:
                ngrams['last_id'] += 1
                ngrams['ngram_freq'][ngram] = {'freq': 0, 'ngram_id': ngrams['last_id'] - 1}
            ngrams['ngram_freq'][ngram]['freq'] += 1
            ngrams['ngram_freq'][ngram]['lemmas'] = ngram_lemmas
            ngrams['tot_ngram_freq'] += 1
    return True


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
            row = [ngram_d['ngram_id'], ' '.join(ngram), ' '.join(ngram_d['lemmas']), ngram_d['freq']]
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
                if type(contexts) is collections.Counter:
                    for context, frequency in contexts.items():
                        row = [col_id, pair_id, pair[0], pair[1], pair_freq, n_type, context, frequency]
                        pattern_writer.writerow(row)
                        col_id += 1
    logging.info('Patterns, saved file: %s.' % outfile_path)


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
    logging.info('Statistics, saved files:\n\t%s' % '\n\t'.join(filenames))


def save_all(wlist_fn: str, nlist_fn: str, plist_fn: str, corpus_fn: str, output_dir: str,
             dump_every=10**5, pickled_inp_dir=None, pickled_out_dir=None, overwrite_pickles=False):
    """Save statistics, patterns and ngram output files in a folder.

    Args:
        wlist_fn: Path to the file with the word list to use for statistics.
        nlist_fn: Path to the file with the word list to use for ngrams.
        plist_fn: Path to the file with the list of pair words to use for the patterns.
        corpus_fn: Path to the file with the corpus.
        output_dir: Path to the output file.
        dump_every: Dump pickle file for ngrams, patterns and stats after the indicated no. of sentences.
        pickled_inp_dir: A folder containing pickled files dictionaries and index.
        pickled_out_dir: Path to folder that stores the pickled files.
        overwrite_pickles: If true, overwrite existing pickles.
    """

    logging.info('Extracting ngrams, patterns and statistics.')
    start_from = 0
    pickled = None
    corpus_len = sum(1 for _ in get_sentences(corpus_fn))

    if pickled_out_dir and any(os.path.exists(join(pickled_out_dir, file))
                               for file in ['ngrams.p', 'statistics.p', 'patterns.p']):
        if overwrite_pickles:
            logging.warning('Pickle files exist in %s. Overwriting them.' % pickled_out_dir)
        else:
            logging.error('Pickles already exists in %s. Use another folder.' % pickled_out_dir)
            return False

    # TODO: split this, make a decorator for picking individual extractions, and add function save_one()
    if pickled_inp_dir:
        pickled_files = (join(pickled_inp_dir, file) for file in ['ngrams.p', 'patterns.p', 'statistics.p'])
        pickled = (pickle.load(open(pickle_file, 'rb')) for pickle_file in pickled_files)
        start_from = pickle.load(open(join(pickled_inp_dir, 'last_sentence_index.p'), 'rb'))
        logging.info('Pickles loaded. Starting at sentence number %s of %s.' % (str(start_from+1), str(corpus_len)))
        corpus_len -= start_from

    ngrams, patterns, statistics = (dict() for _ in range(3)) if not pickled_inp_dir else pickled
    word_list = _get_wlist(wlist_fn)
    ngram_list = _get_wlist(nlist_fn)
    pattern_pairs = _get_pattern_pairs(plist_fn)
    for sentence_no, sentence in enumerate(tqdm.tqdm(get_sentences(corpus_fn), mininterval=0.5, total=corpus_len)):
        if sentence_no < start_from:
            continue
        ngram_args = (sentence, word_list, ngrams)
        pattern_args = (sentence, pattern_pairs, patterns)
        stat_args = (sentence, ngram_list, statistics)
        # Comment out any of the following lines to not run the specified extraction.
        for f, args in (
        #                (extract_ngrams, ngram_args),
                        (extract_patterns, pattern_args),
                        (extract_statistics, stat_args),):
            if not f(*args):
                logger.warning("Function {}() failed:\nsentence: {}".format(f.__name__, sentence))

        if dump_every and not ((sentence_no + 1) % dump_every):
            pickle.dump(ngrams, open(join(pickled_out_dir, 'ngrams.p'), 'wb'))
            pickle.dump(statistics, open(join(pickled_out_dir, 'statistics.p'), 'wb'))
            pickle.dump(patterns, open(join(pickled_out_dir, 'patterns.p'), 'wb'))
            pickle.dump(sentence_no, open(join(pickled_out_dir, 'last_sentence_index.p'), 'wb'))
            logger.info('Pickle files dumped in: %s' % pickled_out_dir)

    logging.info('Extraction completed.')
    # ngrams_prob = add_ngram_probability(ngrams)
    # save_ngrams(ngrams, join(output_dir, 'ngrams.csv'))
    save_patterns(patterns, join(output_dir, 'patterns.csv'))
    save_statistics(statistics, join(output_dir, 'statistics.csv'))
    # save_ngram_stats(ngrams_prob, statistics, join(output_dir, 'ngram_words.csv'))


def test_data():
    """Save ngrams, patterns and statistics to a file using test data."""
    data_dir = os.path.normpath(join(os.path.dirname(__file__), os.pardir + '/data'))
    test_dir = join(data_dir, 'test')
    wlist_fn = join(test_dir, 'wordlist_long.csv')
    nlist_fn = wlist_fn
    plist_fn = join(test_dir, 'patterns.csv')
    corpus_fn = join(test_dir, 'tiny_corpus.csv')
    output_dir = join(data_dir, 'output')
    pickles = join(output_dir, 'pickle')
    save_all(wlist_fn, nlist_fn, plist_fn, corpus_fn, output_dir,
    pickled_out_dir=pickles, dump_every=5000, overwrite_pickles=True)
    # pickled_inp_dir=pickles)


def main():
    test_data()

if __name__ == '__main__':
    main()
