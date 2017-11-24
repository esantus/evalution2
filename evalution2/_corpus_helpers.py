"""Helper functions for creating and processing an eval corpus.

An eval corpus is saved in a tsv file, and has the following structure:

WORD	LEMMA	POS	INDEX	PARENT	DEP
<text id="ukwac:http://observer.guardian.co.uk/osm/story/0,,1009777,00.html">
<s>
Hooligans	hooligan	NNS	1	4	NMOD
"""

import gzip
import os
import re
import warnings

#: Number of fields in the tsv corpus datafile.
CORPUS_FIELDS = 6
#: List of constants spelling out the separators
TOKEN, LEMMA, POS, INDEX, PARENT, DEP = range(0, CORPUS_FIELDS)


def cap_type(word: str):
    """Returns a string describing the capitalization type of word.

    Args:
        word (string): the word to be analyzed

    Returns:
        A string indicating the capitalization of word: 'none', 'all', 'first' or 'others'
    """

    if word.islower():
        return 'none'
    elif word.isupper():
        return 'all'
    elif word.istitle():
        return 'first'
    else:
        return 'others'

        # TODO: a more elegant way
        # functions = [str.islower, str.isupper, str.istitle]
        # for f in functions:
        #    if f(word):
        #        return f.__name__[2:]
        # return 'others'


def extract_ngrams(wordlist, win, stopwords=True, lemma=True, pos=True, dep=True, PLMI=False):
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
            word_freq[word] = 0 if word not in word_freq else (word_freq[word] + 1)
            tot_word_freq += 1

        # Generating all the ngrams
        ngrams = [tuple(sentence[i:i + win]) for i in range(len(sentence) - 1)]

        # Saving the frequency of every ngram
        for ngram in ngrams:
            ngram_freq[ngram] = 0 if ngram not in ngram_freq else (ngram_freq[ngram] + 1)
            tot_ngram_freq += 1

    # For every ngram that was identified
    for ngram in ngram_freq:

        # Calculating the ngrams only for ngrams containing at least one word in wordlist
        if [word for word in set([w for w in ngram.split()]) if word in wordlist] != []:

            # In calculating PPMI, put a cutoff of freq > 3 to avoid rare events to affect the rank
            if ngram_freq[ngram] > 3:
                ngram_prob = float(ngram_freq[ngram]) / tot_ngram_freq

                # Initializing the variable to calculate the probability of components as independent events
                components_prob = 1
                for word in ngram:
                    components_prob *= float(word_freq[word]) / tot_word_freq

            # Ngram probability in PPMI
            colloc[ngram] = math.log(ngram_prob / components_prob)  # PPMI

            # Adaptation to PLMI
            if PLMI == True:
                colloc[ngram] *= ngram_freq[ngram]  # PLMI

    return colloc


def find_word(word, flags=re.IGNORECASE):
    """
    find_word returns all instances of a word (i.e., MWE) in a sentence

    Args:
        word (string): word that need to be searched; it can be a MWE
        flags (re annotations): whether or not case CASE_SENSITIVE

    Returns:
        List of found words
    """

    # Making sure that special characters are interpreted as simple characters?
    re.sub(r"([.*?])", r"\\\1", word)
    # Returning the result of findall
    return re.compile(r'\b({0})\b'.format(word), flags).findall


def get_sentences(corpus_fn: str, check_corpus=False) -> list:
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
    with open_corpus(corpus_fn) as corpus:
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
                        warnings.warn('Invalid line #%d in %s %s' % (line_no, corpus_fn, line))


def open_corpus(corpus_fn, encoding='ISO-8859-2'):
    """Open an eval corpus and return a file reader.

      :param corpus_fn: path to the corpus filename.
      :param encoding: specify the encoding of the file
      :return: a file reader of the corpus.
    """

    if os.path.basename(corpus_fn).endswith('.gz'):
        corpus = gzip.open(corpus_fn, 'r', encoding=encoding)
    else:
        corpus = open(corpus_fn, 'r', encoding=encoding)

    if len(corpus.readline().split("\t")) != CORPUS_FIELDS:
        corpus.close()
        raise ValueError("'%s' is not a valid evalution2 corpus. Use the function "
                         "convert_corpus(corpus) to create an evalution2 corpus" % corpus_fn)
    return corpus


def remove_links(line: str) -> str:
    """Remove links from a string."""

    link_removal = re.compile(r'<[^>]+>')
    return link_removal.sub('', line.strip())


def sort_ngram(colloc):
    """
    __sort_ngram return the sorted dictionary

    Args:
        colloc (dictionary of strings): dictionary containing the ngrams and their frequency

    Returns:
        colloc (dictionary): sorted dictionary
    """

    return sorted([(ngram, colloc[ngram]) for ngram in colloc], key=lambda x: x[1], reverse=True)


def update_collocations(sentence, collocations, word_freq, tot_word_freq, ngram_freq, tot_ngram_freq, wordlist,
                        win, stopwords=True, lemma=True, pos=True, dep=True, PLMI=False):
    """
    Updates the collocations dictionary for every sentence that is passed to it.

    Args:
        sentence (list of tuples): each tuple contains token, lemma, pos, index, parent_index and dep.
        collocations (dict): it contains the updated data.
        wordlist (set of strings): list of words for which we want to extract ngrams.
        win (int): number of words in the ngram.
        stopwords (bool): True if stopwords should be considered, False otherwise.
        lemma (bool): True if the lemmatized ngrams should be extracted, False if tokenized.
        pos (bool): True if the POS should be attached to the tokens/lemmas.
        dep (bool): True if the dep should be attached to the tokens/lemmas.
        PLMI (bool): True to assign PLMI score, False to assign PPMI.

    Returns:
        Returns a dictionary of ngram sets, for every word in wordlist.
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
        word_freq[word] = 0 if word not in word_freq else (word_freq[word] + 1)
        tot_word_freq += 1

    # Generating all the ngrams
    ngrams = [tuple(sentence[i:i + win]) for i in range(len(sentence) - 1)]

    # Saving the frequency of every ngram
    for ngram in ngrams:
        ngram_freq[ngram] = 0 if ngram not in ngram_freq else (ngram_freq[ngram] + 1)
        tot_ngram_freq += 1

    # For every ngram that was identified
    for ngram in ngram_freq:

        # Calculating the ngrams only for ngrams containing at least one word in wordlist
        if [word for word in set([w for w in ngram.split()]) if word in wordlist] != []:

            # In calculating PPMI, put a cutoff of freq > 3 to avoid rare events to affect the rank
            if ngram_freq[ngram] > 3:
                ngram_prob = float(ngram_freq[ngram]) / tot_ngram_freq

                # Initializing the variable to calculate the probability of components as independent events
                components_prob = 1
                for word in ngram:
                    components_prob *= float(word_freq[word]) / tot_word_freq

            # Ngram probability in PPMI
            colloc[ngram] = math.log(ngram_prob / components_prob)  # PPMI

            # Adaptation to PLMI
            if PLMI == True:
                colloc[ngram] *= ngram_freq[ngram]  # PLMI

        return colloc, word_freq, tot_word_freq, ngram_freq, tot_ngram_freq
