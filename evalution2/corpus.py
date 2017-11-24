"""Functions to generate annotated corpora from raw text files.

This module contains a set of functions that are used to generate the annotated gold dataset for model evaluation.

Examples:
    Convert a corpus into an eval corpus, which is a tsv containing the list of annotated words in the corpus.
    >>> corpus = convert_corpus('../data/test_raw')
    Fetch a list of words
    >>> wlist = '..\data\\wl.csv'
    We now create a lemma keyed dictionary which contains info for each entry in wlist.
    >>> pprint((extract_statistics(corpus, wlist)[:2])
    >>> pprint(pattern_search()[:2])
"""

import codecs
from pprint import pprint

# TODO: fix this when packaged properly: from evaluation._corpus_helpers import *
from _corpus_helpers import *


def convert_corpus(corpus_fn, out_fn=None):
    """Converts a raw text file into a tsv file with the columns in CORPUS_FIELDS.

    Converts a raw text file into a tsv file with the following columns:
    token, lemma, pos, index, parent, dep.
    if is saved in a file named "processed_corpus.txt". More columns can be
    defined and added in the future.

    Args:
        corpus_fn (str): path of the corpus.
        out_fn (str): filename of output file. If None, write to $(corpus_fn)_eval.csv.
    Returns: (str)
        Name of the output file.

    """

    with codecs.open(out_fn, "w", "utf-8") as f_corpus:
        # Loading Spacy's parser
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
                f_corpus.write(
                    token.orth_ + "\t" + token.lemma_.lower() + "\t" + token.pos_ + "\t" + str(token.i) + "\t" + (
                        str(token.head.i) if token.head.i != 0 else "0") + "\t" + token.dep_ + "\n")
        f_corpus.write("</s>\n")


def extract_statistics(corpus_fn, wlist_fn, out_fn=None):
    """Extracts statistical information for each word in wordlist and stores it in a dictionary.

    Args:
        corpus_fn (str): path to the evalution2 corpus
        wlist_fn (set of strings): file to be opened and processed.
    Returns:
        A dictionary with infornmation about each lemma. For example.

        >>> pprint(extract_statistics(corpus_fn, wlist_fn))[0]
        'church': {'cap': {'all': 0, 'first': 19, 'none': 169, 'others': 0},
                'freq': 188,
                'norm': {'church': 166, 'churches': 22},
                'pos_dep': {'NNS_COORD': 3,
                            'NNS_OBJ': 4,
                            'NN_VMOD': 1, ...}}
    """

    words = set()
    mwes = list()
    with open(wlist_fn, 'r', encoding='utf-8') as wlist_reader:
        # words = set(line.strip() for line in wlist_reader if len(line.split(' ')) < 2)
        for line in wlist_reader:
            if len(line.split(' ')) < 2:
                words.add(line.strip())
            else:
                mwes.append(tuple(line.strip() for line in line.split(' ')))
        print(mwes)
    with codecs.open(out_fn, "w", "utf-8") as f_statistics:
        # TODO: add support for MWEs
        # Dictionaries for extracting collocations
        ngram_win2_ppmi_slpd = {'word_freq': {}, 'tot_word_freq': 0, 'ngram_freq': {}, 'tot_ngram_freq': 0,
                                'collocations': {}}
        ngram_win3_ppmi_slpd = {'word_freq': {}, 'tot_word_freq': 0, 'ngram_freq': {}, 'tot_ngram_freq': 0,
                                'collocations': {}}
        statistics = {}
        for sentence in get_sentences(corpus_fn):
            pos = 0
            all_tokens = [w[TOKEN] for w in sentence]
            print(all_tokens)
            for word in sentence:
                lemma = word[LEMMA]
                is_mwe = False
                # Check for MWEs
                for mwe in mwes:
                    if lemma == mwe[0]:
                        joined_mwe = ' '.join(mwe)
                        # The window is the seq that goes from the index of the first matched word to the len of the MWE
                        # Notice that MWEs are processed as tokens, not as lemmas.
                        window = ' '.join(all_tokens[pos:pos + len(mwe)])
                        if joined_mwe == window:
                            lemma = joined_mwe
                            pos += len(mwe) - 1
                            is_mwe = True
                            break
                if lemma in words or is_mwe:
                    # TODO: refactor using UserDict.__missing__ and d.update instead of this.
                    print(lemma)
                    if lemma not in statistics:
                        # statistics[lemma] = StatsDict()
                        statistics[lemma] = {}
                        statistics[lemma]["freq"] = 0
                        statistics[lemma]["norm"] = {}
                        statistics[lemma]["cap"] = {cap: 0 for cap in ("first", "all", "none", "others")}
                        statistics[lemma]["pos_dep"] = {}
                    norm_token = lemma if is_mwe else word[TOKEN]
                    # The normalized form of proper nouns is left capitalized.
                    if not lemma.istitle():
                        norm_token = norm_token.lower()
                    if norm_token not in statistics[lemma]["norm"]:
                        statistics[lemma]["norm"][norm_token] = 0
                    if word[POS] + "_" + word[DEP] not in statistics[lemma]["pos_dep"]:
                        statistics[lemma]["pos_dep"][word[POS] + "_" + word[DEP]] = 0

                    statistics[lemma]["freq"] += 1
                    statistics[lemma]["norm"][norm_token] += 1
                    statistics[lemma]["cap"][cap_type(word[TOKEN])] += 1
                    statistics[lemma]["pos_dep"][word[POS] + "_" + word[DEP]] += 1
                    # ngram_win2_ppmi_slpd = update_collocations(sentence, ngram_win2_ppmi_slpd, words, 2,
                    #                                           stopwords=True, lemma=True, pos=True, dep=True,
                    #                                           PLMI=False)
                    # ngram_win3_ppmi_slpd = update_collocations(sentence, ngram_win3_ppmi_slpd, words, 3,
                    #                                           stopwords=True, lemma=True, pos=True, dep=True,
                    #                                           PLMI=False)
                pos += 1
            pprint(statistics)
            break
        # pprint(statistics['church'])
        return statistics


def pattern_search(pattern):
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


def main():
    wlist = '..\data\\wl.csv'
    corpus = '..\data\\corpus.csv'
    extract_statistics(corpus, wlist, 'stats.txt')


main()
