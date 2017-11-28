"""Functions to generate annotated corpora from raw text files.

An eval corpus is saved in a tsv file, and has the following structure:

WORD	LEMMA	POS	INDEX	PARENT	DEP
<text id="ukwac:http://observer.guardian.co.uk/osm/story/0,,1009777,00.html">
<s>
Hooligans	hooligan	NNS	1	4	NMOD

This module further contains a set of functions that are used to generate the annotated gold dataset.

Examples:
    >>>
"""

import collections
import csv
import gzip
import inspect
import logging
import math
import os

import nltk
import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#: Number of fields in the tsv corpus datafile.
CORPUS_FIELDS = 6
#: List of constants spelling out the separators
TOKEN, LEMMA, POS, INDEX, PARENT, DEP = range(0, CORPUS_FIELDS)


class OrderedCounter(collections.Counter, collections.OrderedDict):
    pass


def _cap_type(word: str):
    """Returns a string describing the capitalization type of word.

    Args:
        word (string): the word to be analyzed

    Returns:
        A string indicating the capitalization of word: 'none', 'all', 'first' or 'others'
    """

    functions = [str.islower, str.isupper, str.istitle]
    for f in functions:
        if f(word):
            return f.__name__[2:]
    return 'other'


def _get_pattern_pairs(wlist, separator="\t"):
    """Get a set of unique, symmetric word pairs from a file containing pairs of words.

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
            else:
                logging.warning("line '%s' in corpus '%s' is not a valid pair" % (line, wlist))
    return pattern_pairs


def _get_wlist(wlist_fn):
    """"Generate a set of MWEs and a list of words from a file.

    Args:
        wlist_fn: a file that contains a list of words or MWEs.

    Returns:
        words, mwes: the set of MWEs and a list of words in wlist_fn.
    """
    words = set()
    mwes = list()
    with open(wlist_fn, 'r', encoding='utf-8') as wlist_reader:
        for line in wlist_reader:
            if len(line.split(' ')) < 2:
                words.add(line.strip())
            else:
                mwes.append(list(word.strip() for word in line.split(' ')))
    return words, mwes


def _open_corpus(corpus_fn, encoding='ISO-8859-2'):
    """Open an eval corpus and return a file reader."""

    if os.path.basename(corpus_fn).endswith('.gz'):
        corpus = gzip.open(corpus_fn, 'r', encoding=encoding)
    else:
        corpus = open(corpus_fn, 'r', encoding=encoding)

    if len(corpus.readline().split("\t")) != CORPUS_FIELDS:
        corpus.close()
        raise ValueError("'%s' is not a valid evalution2 corpus. Use the function "
                         "convert_corpus(corpus) to create an evalution2 corpus" % corpus_fn)
    return corpus


def get_sentences(corpus_fn: str, check_corpus=False):
    """
    Yield all the sentences in an eval corpus file.

    Args:
        corpus_fn (str): path to the file containing the corpus.
        check_corpus (bool): if True, print corpus warnings
    Yields:
        a list of tuples representing a sentence in the corpus.
    """

    s = []
    line_no = 1
    with _open_corpus(corpus_fn) as corpus:
        for line in corpus:
            # The header is read in open_corpus, so we start from line 2
            line_no += 1
            # Ignore start and end of DOC
            if '<text' in line or '</text' in line or '<s>' in line:
                continue
            # Yield at the end of SENTENCE
            elif '</s>' in line:
                yield s
                s = []
            # Append all TOKENS
            else:
                s_line = line.split()
                # only append valid lines
                if len(s_line) == CORPUS_FIELDS:
                    word, lemma, pos, index, parent, dep = line.split()
                    s.append((word, lemma, pos, int(index), int(parent), dep))
                else:
                    if check_corpus:
                        logger.warning('Invalid line #%d in %s %s' % (line_no, corpus_fn, line))


def extract_statistics(sentence, words, mwes, statistics):
    """Extracts statistical information for each word in words and mwes and stores it in w_stats.

    The dictionary is strucutred as follows:
    statistics {
        'word': {
            'cap': {lower<str>: freq<int>, upper<str>:freq<int>, title<str>:freq<int>, other<str>:freq<int>
            'freq': freq<int>
            'norm: dict(form<str>: freq<int>): # a dictionary containing normalized forms and their frequencies
            'pos_dep': dict(pos-dep<str>: freq<int>):
    Returns:
        True if the dictionary was updated sucessfully.
    """

    pos = 0
    all_tokens = [w[TOKEN] for w in sentence]
    for word in sentence:
        c_word = word[TOKEN]
        is_mwe = False
        # Check for MWEs
        for mwe in mwes:
            if c_word == mwe[0]:
                joined_mwe = ' '.join(mwe)
                # The window is the seq that goes from the index of the first matched word to the len of the MWE
                window = ' '.join(all_tokens[pos:pos + len(mwe)])
                if joined_mwe == window:
                    c_word = joined_mwe
                    pos += len(mwe) - 1
                    is_mwe = True
                    break
        # MWEs are processed as tokens, singleton as lemmas.
        if not is_mwe:
            c_word = word[LEMMA]
        if c_word in words or is_mwe:
            if c_word not in statistics:
                statistics[c_word] = {}
                statistics[c_word]["freq"] = 0
                statistics[c_word]["cap"] = {cap: 0 for cap in ("lower", "upper", "title", "other")}
                statistics[c_word]["norm"] = collections.Counter()
                statistics[c_word]["pos_dep"] = collections.Counter()
            norm_token = c_word if is_mwe else word[TOKEN]
            # Only the normalized form of proper nouns is left capitalized.
            if not c_word.istitle() or is_mwe:
                norm_token = norm_token.lower()
            statistics[c_word]["freq"] += 1
            statistics[c_word]["norm"][norm_token] += 1
            statistics[c_word]["cap"][_cap_type(word[TOKEN])] += 1
            statistics[c_word]["pos_dep"][word[POS] + "_" + word[DEP]] += 1
        pos += 1
    return True


def extract_patterns(sentence, word_pairs, patterns, islemma=False,
                     save_token=True, save_dep=False, save_lemma=False, save_parent=False, save_pos=False):
    """Returns a dictionary containing all the words between each pair of a set of pair of words.

    The dictionary has the following structure:
    patterns {
            (w1, w1): { # keys are pairs (tuples) of two strings indicating a word (token, or lemma if islemma == True)
            DEP: ... # the values are another dictionary whose keys are the corpus fields (DEP, LEMMA, TOKEN, ...)
            LEMMA: dict(span<str>: freq<int>) # each dictionary contains the in-between span and its frequency.}}
    """

    # TODO: replace with a dictionary of save_ arguments and check if they exist as corpus fields.
    frame = inspect.currentframe()
    args = inspect.getargvalues(frame)[3]
    save_args = {k.upper(): v for (k, v) in args.items() if k.startswith('save_')}
    # First time running, initialize dictionary
    if not patterns:
        for pair in word_pairs:
            # We use OrderedCounter to cross-reference the fields.
            patterns[pair] = {arg[5:].upper(): OrderedCounter() for (arg, val) in save_args.items() if val}
            # Explicitly declare freq = 0 just in case we need to iterate over keys.
            patterns[pair]['freq'] = 0
    # TODO: Benchmark a regexp approach?
    all_words = [w[LEMMA] for w in sentence] if islemma else [w[TOKEN] for w in sentence]
    for pair in word_pairs:
        for i, word in enumerate(all_words):
            if word == pair[0]:
                word1, word2 = pair[0], pair[1]
            elif word == pair[1]:
                word1, word2 = pair[1], pair[0]
            else:
                continue
            match_index = i
            for x in range(i, len(all_words)):
                if all_words[x] == word2:
                    patterns[pair]['freq'] += 1
                    for arg, save in save_args.items():
                        if save:
                            target_name = arg[5:]
                            corpus_field = globals()[target_name]
                            all_targets = [w[corpus_field] for w in sentence]
                            in_between = ' '.join(all_targets[match_index:x + 1])
                            patterns[pair][target_name][in_between] += 1
    return True

def extract_ngrams(sentence, wordlist, ngrams, win=2, include_stopwords=False, islemma=True, pos=True, dep=True,
                   PLMI=False):
    """Extract_ngrams from a sentence and update an ngram dictionary.

    The ngram dictionary has the following structure:
    ngram {
            'tot_ngram_freq':freq<int>: total number of ngrams
            'tot_word_freq': freq<int>: total number of words
            'word_freq': <dict(word: freq<int>)>: a dict containing all the words (in the ngram format) and their freq.
            'ngram_freq': <dict(ngram<tuple>: dict(freq: freq<int>): the key of the dictionary are ngram pairs (tuples),
                the values are dictionary containing only the key 'freq' and the frequency of the ngram.
                The dictionary can be further populated by add_ngram_probability().}
    """

    if not ngrams:
        ngrams.update(tot_word_freq=0, word_freq=collections.Counter(), tot_ngram_freq=0, ngram_freq={})
    # TODO: check if this should include stopwords
    field = LEMMA if islemma else TOKEN
    ngrams['tot_word_freq'] += len(sentence)

    if not include_stopwords:
        # TODO: why stopwords?
        stopwords = nltk.corpus.stopwords.words('english')
        sentence = [w for w in sentence if w[TOKEN] not in stopwords]

    if pos and not dep:
        raw_sentence = [str(w[field] + '-' + w[POS]) for w in sentence]
    elif pos and dep:
        raw_sentence = [str(w[DEP] + ':' + w[field] + '-' + w[POS]) for w in sentence]
    else:
        raw_sentence = [w[field] for w in sentence]

    for word in raw_sentence:
        ngrams['word_freq'][word] += 1
    # Generates the ngrams
    for i, word in enumerate(raw_sentence):
        if sentence[i][field] in wordlist:
            ngram = tuple(raw_sentence[i:i + win])
            # TODO: refactor with defaultdict
            if not ngram in ngrams['ngram_freq']:
                ngrams['ngram_freq'][ngram] = {'freq': 0}
            ngrams['ngram_freq'][ngram]['freq'] += 1
            ngrams['tot_ngram_freq'] += 1
    return True

def add_ngram_probability(ngrams, plmi=False):
    """Add a probability value to each ngram as ngrams[ngram_freq][ngram]['probability']."""
    # For every ngram that was identified
    for ngram in ngrams['ngram_freq']:
        # In calculating PPMI, put a cutoff of freq > 3 to avoid rare events to affect the rank
        curr_ngram = ngrams['ngram_freq'][ngram]
        if curr_ngram['freq'] > 3:
            ngram_prob = float(curr_ngram['freq']) / ngrams['tot_ngram_freq']
            # Initializing the variable to calculate the probability of components as independent events
            components_prob = 1
            for word in ngram:
                components_prob *= float(ngrams['word_freq'][word]) / ngrams['tot_word_freq']
            # ngram probability in PPMI
            probability = math.log(ngram_prob / components_prob)  # PPMI
            # Adaptation to PLMI
            if plmi:
                probability *= curr_ngram['freq']  # PLMI
            ngrams['ngram_freq'][ngram]['probability'] = probability
    return ngrams


# TODO: Maybe use a decorator.
def save_ngrams(ngrams, outfile_path, probability=True):
    """Save ngrams to a tsv file."""

    if len(ngrams) != 4:
        raise ValueError('@param ngrams must be a valid evalution ngram dictionary.')

    with open(outfile_path, 'w', encoding='utf-8', newline='') as outfile:
        ngram_writer = csv.writer(outfile)
        header = ['id', 'ngram', 'freq']
        if probability:
            header.append('probability')
        ngram_writer.writerow(header)
        for col_id, ngram_tuple in enumerate(ngrams['ngram_freq']):
            ngram = ngrams['ngram_freq'][ngram_tuple]
            row = [col_id, ngram_tuple, ngram['freq']]
            if probability:
                row.append(ngram.get('probability', 'NA'))
            ngram_writer.writerow(row)
    logging.info('%s saved.' % outfile_path)


def save_patterns(patterns, outfile_path):
    """Save patterns dictionary in a csv file."""

    if len(patterns) != 2:
        raise ValueError('@param ngrams must be a valid evalution ngram dictionary.')

    with open(outfile_path, 'w', encoding='utf-8', newline='') as outfile:
        pattern_writer = csv.writer(outfile)
        header = ['id', 'pair', 'pair freq.', 'type', 'context', 'frequency']
        pattern_writer.writerow(header)
        col_id = 0
        for pair, ntypes in patterns.items():
            pair_freq = ntypes['freq']
            for ntype, contexts in ntypes.items():
                if ntype != 'freq':
                    for context, frequency in contexts.items():
                        row = [col_id, pair, pair_freq, ntype, context, frequency]
                        pattern_writer.writerow(row)
                        col_id += 1
    logging.info('%s saved.' % outfile_path)


def save_statistics(statistics, outfile_path):
    """Save statistics dictionary in a csv file."""

    if len(statistics) != 11:
        raise ValueError('@param ngrams must be a valid evalution ngram dictionary.')

    # TODO: refactor this trash.
    filenames = ["{}_{}.csv".format(os.path.splitext(outfile_path)[0], suffix)
                 for suffix in ['words', 'forms', 'posdep']]
    open_args = dict(mode='w', encoding='utf-8', newline='')
    with open(filenames[0], **open_args) as outfile_words, open(
            filenames[1], **open_args) as outfile_norm, open(
        filenames[2], **open_args) as outfile_posdep:

        wordf = csv.writer(outfile_words)
        normf = csv.writer(outfile_norm)
        posdepf = csv.writer(outfile_posdep)
        header_main = ['id', 'word', 'freq.', 'cap_lower', 'cap_upper', 'cap_title', 'cap_other']
        header_norm = ['id', 'word_id', 'norm.', 'freq.']
        header_posdep = ['id', 'word_id', 'posdep', 'freq.']
        wordf.writerow(header_main)
        normf.writerow(header_norm)
        posdepf.writerow(header_posdep)
        word_id, norm_id, posdep_id = (0 for _ in range(3))

        for word, attrs in statistics.items():
            cap = attrs['cap']
            # TODO: make cap ordered and then unpack.
            row = [word_id, word, cap['lower'], cap['upper'], cap['title'], cap['other']]
            wordf.writerow(row)
            for attr_name, attr_values in attrs.items():
                if attr_name == 'norm':
                    for norm, freq in attr_values.items():
                        row = [norm_id, word_id, norm, freq]
                        normf.writerow(row)
                        norm_id += 1
                elif attr_name == 'pos_dep':
                    for posdep, freq in attr_values.items():
                        row = [posdep_id, word_id, posdep, freq]
                        posdepf.writerow(row)
                        posdep_id += 1
            word_id += 1
    logging.info('%s saved.' % ', '.join(filenames))


def main():
    wlist = '..\data\\test\\wl.csv'
    corpus = '..\data\\test\\tiny_corpus.csv'
    patterns_fn = '..\data\\test\\patterns.txt'
    ngrams, patterns, statistics = (dict() for _ in range(3))
    words, mwes = _get_wlist(wlist)
    pattern_pairs = _get_pattern_pairs(patterns_fn)

    logging.info('Extracting ngrams, patterns and statistics.')
    for sentence in tqdm.tqdm(get_sentences(corpus), mininterval=0.5):
        ngram_args = (sentence, words, ngrams)
        pattern_args = (sentence, pattern_pairs, patterns, 0, 1, 1, 1)
        stat_args = (sentence, words, mwes, statistics)
        for f, args in ((extract_ngrams, ngram_args),
                        (extract_patterns, pattern_args),
                        (extract_statistics, stat_args)):
            if not f(*args):
                logger.warning("Function {}() failed:\nsentence: {}".format(f.__name__, sentence))
    logging.info('Extraction completed.')
    ngrams = add_ngram_probability(ngrams)
    output_dir = '..\\data\\output\\'
    save_ngrams(ngrams, output_dir + 'ngrams.csv')
    save_patterns(patterns, output_dir + 'patterns.csv')
    save_statistics(statistics, output_dir + 'statistics.csv')
    # pprint(patterns)
    # pprint(statistics)
    # pprint(statistics['church'])
    # pprint(statistics['used to be'])


if __name__ == '__main__':
    main()