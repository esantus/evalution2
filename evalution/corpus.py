# coding=utf-8
"""Functions to generate annotated corpora from raw text files and to extract data from a corpus.

Example:

    The core of this module is constituted by the three extract functions:
        extract_ngrams(): extract ngrams from a corpus given a set of words.
        extract_patterns(): extract patterns from a corpus given a set of word pairs.
        extract_frequencies(): extract word frequencies from a corpus given a set of words.

First, we extract a list of words (get_wlist) or pair of words (get_pattern_list()).

>>> wlist_fn = join('..', 'data', 'test',  'wordlist_long.csv')
>>> wlist = get_wlist(wlist_fn)

We can use the word list to initialize a Dataset instance (or to coll the add_ functions directly).

>>> dataset = Dataset(wlist)

The dataset class contains three dicionaries which will hold the extracted data.

>>> dataset.ngrams
>>> dataset.patterns
>>> dataset.frequencies

We then iterate through the corpus sentence by sentence using get_sentences(filename).

>>> corpus_fn = join('..', 'data', 'test',  'corpora', 'tint_corpus.csv')
>>> for sentence_no, sentence in enumerate(get_sentences(corpus_fn):
...     dataset.add_patterns(sentence, sentence_no)
...     dataset.add_frequencies(sentence, sentence_no)

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
"""

import collections
import csv
import gzip
import logging
import math
import os
import pickle
import pprint
import re
try:
    import reprlib
except ImportError:
    import repr as reprlib

from os.path import join
from typing import AnyStr, List, Mapping, MutableMapping, Set, Sequence, TextIO, Tuple

import tqdm
from flashtext import KeywordProcessor

from evalution import data

logger = logging.getLogger(__name__)
#: Corpus fields in an eval corpus
CORPUS_FIELDS = ['token', 'lemma', 'pos', 'index', 'parent', 'dep', 'lemma_i', 'token_i']
PatternList = Set[Tuple[str, str]]
Word = collections.namedtuple('Word', CORPUS_FIELDS)


def _cap_type(word: AnyStr) -> str:
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


def _is_stopword(word: AnyStr) -> bool:
    """Returns true if a word is a stopword."""
    if word in data.stopwords or word.endswith("'ll") or word.endswith("'t"):
        return True
    return False


def _open_corpus(corpus_path: AnyStr, encoding: AnyStr='ISO-8859-2') -> TextIO:
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

    for filename in to_open:
        if os.path.basename(filename).endswith('.gz'):
            corpus = gzip.open(filename, 'rt', encoding=encoding)
        else:
            corpus = open(filename, encoding=encoding)
        if len(corpus.readline().split("\t")) != len(CORPUS_FIELDS) - 2:
            corpus.close()
            raise ValueError("'%s' is not a valid evalution2 corpus. Try convert_corpus(corpus)." % filename)
        yield corpus


def get_pattern_pairs(wlist: AnyStr, separator: AnyStr = "\t") -> PatternList:
    """Get a set of unique, symmetric word pairs from a file containing pairs of words.

    Args:
        wlist: Path to the file containing the list of pairs of words to fetch.
        separator: The separator used in the wlist file.

    Returns:
        A set of pairs and their inverse. For example: {('the', 'a'), ('a', 'the'), ('for', 'be'), ('be', 'for')}
    """

    pattern_pairs = set()
    with open(wlist) as pattern_reader:
        for line in pattern_reader:
            split_line = tuple(word.strip() for word in line.split(separator))
            if len(split_line) == 2:
                if not any(pair in pattern_pairs for pair in (split_line, split_line[::-1])):
                    pattern_pairs.add(split_line)
                    pattern_pairs.add(split_line[::-1])
            else:
                logging.warning("line '%s' in corpus '%s' is not a valid pair" % (line, wlist))
    return pattern_pairs


def get_wlist(wlist_fn: AnyStr) -> KeywordProcessor:
    """"Generate a set of MWEs and a list of words from a file.

    Args:
        wlist_fn: A file that contains a list of words or MWEs.

    Returns:
        A KeywordProcessor containing the list of words from wlist_fn.
    """

    words = KeywordProcessor(case_sensitive=True)
    words.non_word_boundaries.add('-')
    words.non_word_boundaries.add('\'')
    # words.add_keyword_from_file(wlist_fn)
    with open(wlist_fn, encoding='utf-8') as wlist_reader:
        for line in wlist_reader:
            words.add_keyword(line.strip())
    return words


class Sentence:
    __slots__ = 'words',

    def __init__(self, words: List[Word]):
        """A sentence composed of several annotated words (Word)."""
        self.words = words

    def __getattr__(self, item):
        return [w for w in self.words if w.lemma == item]

    def __getitem__(self, position):
        return self.words[position]

    def __len__(self):
        return len(self.words)

    def __str__(self):
        return 'Sentence(%s)' % reprlib.repr(self.tokens)

    def __repr__(self):
        return pprint.pprint(self.words)

    @property
    def tokens(self):
        return ' '.join([word.token for word in self.words])


# noinspection PyUnresolvedReferences
class Dataset:
    def __init__(self, **kwargs) -> None:
        """Dataset object containing ngrams, frequencies and pattern dictionaries, and methods to extract and save them.

        Args:
            w_list: set of words to search the frequencies for.
            n_list: set of words to use to extract the ngrams.
            p_list: set of word pairs to use to extract the patterns.
            pickle_every: Dump pickle file for ngrams, patterns and stats after the indicated no. of sentences.
            pickle_out: Path to folder that stores the pickled files.
            overwrite_pickles: if set to False, raise a warning when trying to write on a folder with existing pickles.
        """

        self.__dict__.update(**kwargs)
        self._pickle_names = ('ngrams.p', 'patterns.p', 'frequencies.p')
        self._overwrite_pickles = None
        self.start_from = 0
        self.pickled = None
        self.ngrams = NgramCollection()
        self.patterns = {}
        self.frequencies = {}
        self.overwrite_pickles = kwargs['overwrite_pickles']

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
                if instance.pickle_every \
                        and instance.overwrite_pickles \
                        and not (sentence_no + 1) % instance.pickle_every:
                    # We need to use a new protocol to pickle objects with __slots__
                    pickle.dump(getattr(instance, self.to_pickle),
                                open(join(instance.pickle_out, self.pickle_file), 'wb'), protocol=4)
                    logging.info('%s pickled in: %s' % (self.pickle_file, instance.pickle_out))
            return pickled_add

    @Pickler(to_pickle='ngrams')
    def add_ngrams(self, sentence):
        extract_ngrams(sentence, self.n_list, self.ngrams)

    @Pickler(to_pickle='patterns')
    def add_patterns(self, sentence):
        extract_patterns(sentence, self.p_list, self.patterns)

    @Pickler(to_pickle='frequencies')
    def add_frequencies(self, sentence):
        extract_frequencies(sentence, self.w_list, self.frequencies)

    def add_ngram_prob(self):
        self.ngrams = add_ngram_probability(self.ngrams)

    def load_pickles(self, pickle_names):
        # Assuming it is a folder. Not a good idea to ask for forgiveness here.
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
                logging.info('Found last_index.p. Resuming from sentence number ' + str(self.start_from + 1))
        else:
            try:
                pickles = pickle_names[:3]
            except:
                raise TypeError('pickle_names must be a string or a sequence of len 4.')
            else:
                self.start_from = pickles[3] if pickles[3] else 0

        self.ngrams, self.patterns, self.frequencies = (pickle.load(open(pickle_file, 'rb')) for pickle_file in pickles)
        logging.info('Pickles loaded.')

    def save_all(self, output_dir=os.getcwd()):
        if self.ngrams:
            save_ngrams(self.ngrams, join(output_dir, 'ngrams.csv'))
            if self.frequencies:
                save_ngram_stats(self.ngrams, self.frequencies, join(output_dir, 'ngram_words.csv'))
        if self.patterns:
            save_patterns(self.patterns, join(output_dir, 'patterns.csv'))
        if self.frequencies:
            save_frequencies(self.frequencies, join(output_dir, 'frequencies.csv'))


class WordFrequencies:
    __slots__ = ('word', 'id', 'freq', 'cap', 'norm', 'pos_dep')

    def __init__(self, word: AnyStr, stat_id: int=None):
        """A __slots__ class representing a word and its frequency information.

        Attributes:
            self.word: The lemma of the word we get the frequencies for.
            self.id: The id of the lemma.
            self.freq: The frequency of the word.
            self.cap: A dictionary containing the frequency of the word in 'lower', 'upper', 'title', and 'other' form.
            self.norm: A counter with the frequnencies of the word in its normalized forms.
            self.pos_dep: A counter with the frequencies of the word's POS and DEP.
        """

        self.word = word
        self.id = stat_id
        self.freq = 0
        self.cap = {cap: 0 for cap in ("lower", "upper", "title", "other")}
        self.norm = collections.Counter()  # type: MutableMapping[str, int]
        self.pos_dep = collections.Counter()  # type: MutableMapping[str, int]

    def __str__(self):
        return self.word


class PatternFrequencies:
    __slots__ = ('pair', 'id', 'freq', 'token', 'lemma', 'pos', 'dep')

    def __init__(self, pair: Tuple[str, str], pair_id: int = None):
        """A _slots__ class representing a pattern and its frequency information.

        Attribues:
            self.pair: The pair of words.
            self.id: Id of the pair.
            self.freq: The frequency of the pair.
            self.token, self.lemma, self.pos, self.dep:
                The items between the words in the pair represented as token, lemma, pos or dep form and their freq.
            self.fields: Yield self.token, self.lemma, self.pos and self.dep and their name.
        """

        self.pair = pair
        self.id = pair_id
        self.freq = 0
        self.token, self.lemma, self.pos, self.dep = (collections.Counter()
                                                      for _ in range(4))  # type: MutableMapping[str, int]

    def __repr__(self):
        return '%s((%s, %s), %s)' % (self.__class__.__name__, self.pair[0], self.pair[1], str(self.id))

    @property
    def fields(self):
        yield ('token', self.token)
        yield ('lemma', self.lemma)
        yield ('pos', self.pos)
        yield ('dep', self.dep)


class Ngram:
    __slots__ = ('ngram', 'id', 'lemmas', 'freq', 'probability')

    def __init__(self, ngram: Sequence[AnyStr], ngram_id: int=None):
        """A __slots__ class representing an ngram.

        Args:
            ngram: The elements of the ngram.
        """

        self.ngram = ngram
        self.id = ngram_id
        self.lemmas = None
        self.freq = 0
        self.probability = None

    def __repr__(self):
        return '%s%s' % (self.__class__.__name__, repr(self.ngram))

    def __len__(self):
        return len(self.ngram)


class NgramCollection:
    def __init__(self, ngrams: Ngram=None):
        """Includes n-grams as Ngram objects and n-gram related frequencies in a corpus.

        Args:
            ngrams: a collection of Ngram tuples.

        Attributes:
            self.tot_word_freq: Number of processed words.
            self.tot_ngram_freq: Number of ngram found.
            self.word_freq: Frequency of the occurrence of a word in any ngram.
            self.ngrams: A dictionary containing the ngram string as key and an ngram object as value.
        """

        self.last_id = 0
        self.tot_word_freq = 0
        self.tot_ngram_freq = 0
        self.word_freq = collections.Counter()
        self.ngrams = ngrams if ngrams else {}

    def __iter__(self):
        for ngram in self.ngrams.values():
            yield ngram

    def __len__(self):
        return len(self.ngrams)


def get_sentences(corpus_fn: AnyStr, file_encoding: AnyStr = 'utf-8') -> Sentence:
    """
    Yield all the sentences in an eval corpus file as a list of Word namedtuples.

    Args:
        corpus_fn: Filename of the corpus.
        file_encoding: Specify encoding of the corpus.

    Yields:
        A Sentence object.
    """

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
                    yield Sentence(sentence)
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
                            yield Sentence(sentence)
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


def extract_frequencies(sentence: Sentence, words: KeywordProcessor, frequencies: MutableMapping) -> MutableMapping:
    """Extracts statistical information for each word in words and mwes and stores it in w_stats.

    Args:
        sentence: A Sentence object.
        words: KeywordProcessor containing the list of words to count (see _get_wlist()).
        frequencies: A dictionary with the frequencies to update. Keys are lemma, values are WordFrequency objects.

    Returns:
        Reference to the updated dictionary.
    """

    if not frequencies:
        frequencies['last_id'] = 0
    all_lemmas = ' '.join([word.lemma for word in sentence])
    matched_indexes = [match[1] for match in words.extract_keywords(all_lemmas, span_info=True)]
    matched_lemmas = [w for w in sentence if w.lemma_i in matched_indexes]
    for word in matched_lemmas:
        lemma = word.lemma
        if lemma not in frequencies:
            frequencies['last_id'] += 1
            frequencies[lemma] = WordFrequencies(lemma, frequencies['last_id'])
        norm_token = word.token
        # Only the normalized form of proper nouns is left capitalized.
        if not word.token.istitle():
            norm_token = norm_token.lower()
        frequencies[lemma].freq += 1
        frequencies[lemma].norm[norm_token] += 1
        frequencies[lemma].cap[_cap_type(word.token)] += 1
        frequencies[lemma].pos_dep[word.pos + "_" + word.dep] += 1
    return frequencies


def extract_patterns(sentence: Sentence, word_pairs: PatternList, patterns: MutableMapping,
                     save_args: Mapping = None) -> MutableMapping:
    """Returns a dictionary containing all the words between each pair of a set of pair of words.

    Args:
        sentence: A Sentence object.
        word_pairs: A set of word pairs and their inverse.
        patterns: The pattern dictionary to be updated or initialized.
            They keys are string indicating the pair, the values are PatternFrequencies objects.
        save_args: A dictionary with keys indicating a corpus field. If key is true. save the pattern indicated by it.
            Default is all True.
    Returns:
        Reference to the updated dictionary.
    """

    if not save_args:
        save_args = {'token': True, 'lemma': True, 'pos': True, 'dep': True}

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
                    if pair not in patterns:
                        pair_id = 0
                        patterns[pair] = (PatternFrequencies(pair, pair_id))
                    patterns[pair].freq += 1
                    for field, value in save_args.items():
                        if value:
                            all_targets = [getattr(w, field) for w in sentence]
                            in_between = ' '.join(all_targets[match_index + 1:x])
                            if not in_between:
                                # Nothing in between, set a special character #
                                in_between = '#'
                            # Returns a counter with the actual value or 0 if it does not exist.
                            pattern_field = getattr(patterns[pair], field)
                            # We increase the value and reassign.
                            pattern_field[in_between] += 1
                            setattr(patterns[pair], field, pattern_field)
    return patterns


def extract_ngrams(sentence: Sentence, words: KeywordProcessor, ngram_collection: NgramCollection(), win: int = 2,
                   exclude_stopwords: bool = True, istoken: bool = False,
                   pos: bool = True, dep: bool = True) -> MutableMapping:
    """Extract_ngrams from a sentence and update an ngram dictionary.

    Args:
        sentence: A Sentence object.
        words: KeywordProcessor containing the list of words to count (see _get_wlist()).
        ngram_collection: an NgramCollection object: if populated, the ngrams will be updated.
        win: The ngram window.
        exclude_stopwords: if True, exclude stopwords from ngrams.
        istoken: If True, match the tokens instead of lemmas.
        pos: If True, ngram includes POS information.
        dep: If true, ngram includes DEP information.

    Returns:
        Reference to the updated dictionary.
    """

    field = 'token' if istoken else 'lemma'
    lemmas_to_search = ' '.join([word.lemma for word in sentence[:-win + 1]])
    ngram_collection.tot_word_freq += len(sentence)
    matches = collections.deque([match for match in words.extract_keywords(lemmas_to_search, span_info=True)
                                 if exclude_stopwords and not _is_stopword(match[0])])
    i = 0
    last_is_stopword = False
    while matches:
        if i == len(sentence):
            if last_is_stopword:
                return ngram_collection
            # We reached the end, but there are still items in the deque.
            # logging.warning("Missing index for %s" % repr(matches[0]))
            matches.popleft()
            i = 0
            continue
        _, match_begin, match_end = matches[0]
        word = sentence[i]
        if word.lemma_i == match_begin:
            for j in range(i + 1, len(sentence)):
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
                                                for w in ngram_first_slice)),)
                        ngram += tuple((str(w.dep + ':' + str(getattr(w, field)) + '-' + w.pos)
                                        for w in ngram_second_slice if w.lemma not in data.stopwords))
                    else:
                        ngram = tuple(' '.join(getattr(w, field) for w in ngram_first_slice))
                        ngram += tuple(getattr(w, field) for w in ngram_second_slice
                                       if w.lemma not in data.stopwords)
                    ngram_lemmas = ' '.join([w.lemma for w in sentence[i:ngram_end_index]
                                             if w.lemma not in data.stopwords])

                    if ngram not in ngram_collection.ngrams:
                        ngram_collection.ngrams[ngram] = Ngram(ngram, ngram_collection.last_id)
                        ngram_collection.last_id += 1
                    ngram_collection.ngrams[ngram].freq += 1
                    ngram_collection.ngrams[ngram].lemmas = ngram_lemmas
                    ngram_collection.tot_ngram_freq += 1
                    for item in ngram:
                        ngram_collection.word_freq[item] += 1
                    matches.popleft()
                    last_is_stopword = False
                    i = after_target_i - 1
                    break
        i += 1
    return ngram_collection


def add_ngram_probability(ngram_collection: NgramCollection, plmi: bool = False) -> NgramCollection:
    """Add a probability value to each ngram.

    Args:
        ngram_collection: The ngram dictionary to update.
        plmi: Use plmi.

    Returns
        The updated ngram dictionary containing a probability field for each ngram with freq. > 3.
    """

    for ngram in ngram_collection:
        # In calculating ppmi, put a cutoff of freq > 3 to avoid rare events to affect the rank
        if ngram.freq < 4:
            probability = 0
        else:
            ngram_prob = float(ngram.freq) / ngram_collection.tot_ngram_freq
            # Initializing the variable to calculate the probability of components as independent events
            components_prob = 1
            for word in ngram.ngram:
                components_prob *= float(ngram_collection.word_freq[word]) / ngram_collection.tot_word_freq
            probability = math.log(ngram_prob / components_prob)  # ppmi
            if plmi:
                probability *= ngram.freq  # plmi
        ngram_collection.ngrams[ngram.ngram].probability = probability
    return ngram_collection


def save_ngrams(ngram_collection: NgramCollection, outfile_path: AnyStr) -> None:
    """Save ngrams to a tsv file.

    Args:
        ngram_collection: The ngram collection.
        outfile_path: The filename of the output file.
    """

    # save probability only if at least one element has probability > 0
    with open(outfile_path, 'w', encoding='utf-8', newline='') as outfile:
        ngram_writer = csv.writer(outfile)
        header = ['ngram_id', 'ngram', 'lemmas', 'freq', 'probability']
        ngram_writer.writerow(header)
        for ngram in ngram_collection:
            row = [ngram.id, ' '.join(ngram.ngram), ngram.lemmas, ngram.freq, ngram.probability]
            ngram_writer.writerow(row)
    logging.info('%s saved.' % outfile_path)


def save_ngram_stats(ngram_collection: NgramCollection, frequencies: Mapping, outfile_path: AnyStr) -> None:
    """Save a file mapping ngrams_id to word_ids.

    Args:
        ngram_collection: The ngram collection.
        frequencies: The frequencies dictionary.
        outfile_path: The filename of the output file.

    Returns:
        True if file is successfully written.
    """

    with open(outfile_path, 'w', encoding='utf-8', newline='') as outfile:
        ngram_writer = csv.writer(outfile)
        header = ['ngram_id', 'word_id', 'ngram_index']
        ngram_writer.writerow(header)
        for ngram in ngram_collection:
            for ngram_index, lemma in enumerate(ngram.lemmas):
                if lemma in frequencies:
                    word_id = frequencies[lemma].id
                    row = [ngram.id, word_id, ngram_index]
                    ngram_writer.writerow(row)
    logging.info('%s saved.' % outfile_path)


def save_patterns(patterns: MutableMapping, outfile_path: AnyStr) -> None:
    """Save patterns dictionary in a csv file.

    Args:
        patterns: The patterns dictionary.
        outfile_path: The filename of the output file.
    """

    with open(outfile_path, 'w', encoding='utf-8', newline='') as outfile:
        pattern_writer = csv.writer(outfile)
        header = ['pattern_id', 'pair_id', 'word1', 'word2', 'freq.', 'type', 'context', 'frequency']
        pattern_writer.writerow(header)
        col_id = 0
        for pair, frequencies in patterns.items():
            for field_name, field_value in frequencies.fields:
                for context, frequency in field_value.items():
                    row = [col_id, frequencies.id, pair[0], pair[1], frequencies.freq, field_name, context, frequency]
                    pattern_writer.writerow(row)
                    col_id += 1
    logging.info('%s saved.' % outfile_path)


def save_frequencies(frequencies: MutableMapping, outfile_path: AnyStr) -> None:
    """Save frequencies dictionary in a csv file.

    Args:
        frequencies: The frequencies dictionary.
        outfile_path: The filename of the output file.
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

        for word, word_stat in frequencies.items():
            if word == 'last_id':
                continue
            cap = word_stat.cap
            row = [word_id, word, word_stat.freq, cap['lower'], cap['upper'], cap['title'], cap['other']]
            word_f.writerow(row)
            for norm, freq in word_stat.norm.items():
                row = [norm_id, word_id, norm, freq]
                norm_f.writerow(row)
                norm_id += 1
            for posdep, freq in word_stat.pos_dep.items():
                row = [posdep_id, word_id, posdep, freq]
                posdep_f.writerow(row)
                posdep_id += 1
            word_id += 1
    for filename in filenames:
        logging.info('%s saved.' % filename)


def test_data() -> None:
    """Save ngrams, patterns and frequencies to a file using test data."""
    data_dir = os.path.normpath(join(os.path.dirname(__file__), os.pardir + '/data'))
    test_dir = join(data_dir, 'test')
    output_dir = join(data_dir, 'output')
    pickle_dir = join(output_dir, 'pickle')

    wlist_fn = join(test_dir, 'wordlist_long.csv')
    plist_fn = join(test_dir, 'patterns.csv')
    corpus_fn = join(test_dir, 'tiny_corpus.csv')

    nlist_fn = wlist_fn
    dataset = Dataset(w_list=get_wlist(wlist_fn), n_list=get_wlist(nlist_fn), p_list=get_pattern_pairs(plist_fn),
                      pickle_every=5000, pickle_out=pickle_dir, overwrite_pickles=True)
    # dataset.load_pickles(pickle_dir)
    corpus_len = sum(1 for _ in get_sentences(corpus_fn))
    for sentence_no, sentence in enumerate(tqdm.tqdm(get_sentences(corpus_fn), mininterval=0.5, total=corpus_len)):
        dataset.add_ngrams(sentence, sentence_no)
        dataset.add_patterns(sentence, sentence_no)
        dataset.add_frequencies(sentence, sentence_no)
    dataset.add_ngram_prob()
    # pickle.dump(dataset, open('dataset.p', 'wb'), protocol=4)
    dataset.save_all(output_dir)


def main() -> None:
    test_data()


if __name__ == "__main__":
    # To run as a script use python -m evalution.corpus from parent folder.
    main()
