import os
from os.path import join

import pytest

from evalution import corpus as c


# Using this instead of fixtures for now.
class Paths:
    def __init__(self):
        self.data_dir = os.path.normpath(join(os.path.dirname(__file__), os.pardir + '/data'))
        self.test_dir = join(self.data_dir, 'test')
        self.output_dir = join(self.data_dir, 'output')
        self.pickle_dir = join(self.output_dir, 'pickle')
        self.corpora = join(self.test_dir, 'corpora')

        self.wlist_fn = join(self.test_dir, 'wordlist_long.csv')
        self.plist_fn = join(self.test_dir, 'patterns.csv')
        self.corpus_fn = join(self.test_dir, 'tiny_corpus.csv')

        self.nlist_fn = self.wlist_fn

paths = Paths()


@pytest.mark.parametrize("input_str, expected", [
    ("test", "lower"),
    ("Test", "title"),
    ("TEST", "upper"),
    ("TeSt", "other")])
def test__cap_type(input_str, expected):
    # noinspection PyProtectedMember
    assert c._cap_type(input_str) == expected


@pytest.mark.parametrize("input_str, expected", [
    ("a", True),
    ("I'll", True),
    ("Ain't", True),
    ("aplomb", False)])
def test__is_stopword(input_str, expected):
    # noinspection PyProtectedMember
    assert c._is_stopword(input_str) == expected


@pytest.mark.parametrize("path", [
    join(paths.test_dir, 'tiny_corpus.csv'),
    join(paths.corpora, 'multiple_gz'),
    join(paths.corpora, 'multiple_csv')])
def test__open_corpus_pass(path):
    # noinspection PyProtectedMember
    for corpus_reader in c._open_corpus(path):
        with corpus_reader as corpus:
            assert corpus.readline().startswith('<text')


def test__open_corpus_invalid():
    with pytest.raises(ValueError):
        # noinspection PyProtectedMember
        _ = list(c._open_corpus(join(paths.test_dir, 'patterns.csv')))


def test__open_corpus_invalid_path():
    with pytest.raises(FileNotFoundError):
        # noinspection PyProtectedMember
        _ = list(c._open_corpus(join(paths.test_dir, 'not_exist.csv')))


def test_get_pattern_pairs():
    expected = {
        ('animal', 'dog'),
        ('animal', 'dogs'),
        ('army', 'the'),
        ('dog', 'animal'),
        ('dogs', 'animal'),
        ('the', 'army')
    }
    assert expected == c.get_pattern_pairs(paths.plist_fn)


def test_get_wlist():
    assert 'the' in c.get_wlist(paths.wlist_fn)


def test_get_wlist_invalid_word():
    assert 'lksdjfs' not in c.get_wlist(paths.wlist_fn)


def test_get_sentences():
    assert sum(1 for _ in c.get_sentences(paths.corpus_fn)) == 6071

    # Verify that each returned object is a Sentence
    for sentence in c.get_sentences(paths.corpus_fn):
        assert isinstance(sentence, c.Sentence)


def test_get_sentences_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        _ = list(c.get_sentences('nonexistent_corpus.csv'))


def test_frequencies(tmpdir):
    # Load up test data to populate an frequencies dictionary
    corpus_fn = join(paths.test_dir, 'nano_corpus.csv')
    words = c.get_wlist(paths.wlist_fn)
    word_frequencies = {}

    # Extract the frequencies
    for sentence in c.get_sentences(corpus_fn):
        c.extract_frequencies(sentence, words, word_frequencies)

    # Convert the WordFrequency objects into the numeric values for easy
    # comparision
    frequencies = {word: v.freq for word, v in word_frequencies.items() if word != 'last_id'}
    expected = {
        'dog': 3,
        'is': 2,
        'an': 2,
        'animal': 2,
        'such': 1,
    }
    assert frequencies == expected

    # Save the frequencies
    p = tmpdir.mkdir('tmp')
    c.save_frequencies(word_frequencies, p)

    with open(p + '_words.csv') as fh:
        contents = fh.readlines()

    expected_header = 'stat_id,word,freq.,cap_lower,cap_upper,cap_title,cap_other\n'

    assert len(contents) == 6
    assert contents[0] == expected_header
    assert contents[1] == '0,dog,3,2,0,1,0\n'
    assert contents[2] == '1,is,2,2,0,0,0\n'
    assert contents[3] == '2,an,2,2,0,0,0\n'
    assert contents[4] == '3,animal,2,1,0,1,0\n'
    assert contents[5] == '4,such,1,1,0,0,0\n'


def test_patterns(tmpdir):
    corpus_fn = join(paths.test_dir, 'nano_corpus.csv')
    word_pairs = c.get_pattern_pairs(paths.plist_fn)
    patterns = {}

    for sentence in c.get_sentences(corpus_fn):
        c.extract_patterns(sentence, word_pairs, patterns)

    # Convert the PatternFrequency objects into the numeric values for easY comparision
    frequencies = {word: v.freq for word, v in patterns.items()}
    expected = {
        ('dog', 'animal'): 2,
        ('animal', 'dog'): 1,
        ('dogs', 'animal'): 1,
    }
    assert frequencies == expected

    # Save the patterns
    p = tmpdir.join('patterns_output.txt')
    c.save_patterns(patterns, p)

    contents = p.readlines()
    expected_header = ','.join(['pattern_id', 'pair_id', 'word1', 'word2', 'freq.', 'type', 'context', 'frequency']) + '\n'

    assert len(contents) == 14
    assert contents[0] == expected_header
    assert contents[1] == '0,0,dog,animal,2,token,was an,1\n'
    assert contents[2] == '1,0,dog,animal,2,token,is an,1\n'
    assert contents[-1] == '12,0,animal,dog,1,dep,NMOD NMOD,1\n'


def test_ngrams(tmpdir):
    # Load up test data to populate the NgramsCollection
    corpus_fn = join(paths.test_dir, 'nano_corpus.csv')
    words = c.get_wlist(paths.wlist_fn)
    ngrams = c.NgramCollection()

    # Extract the ngrams
    for sentence in c.get_sentences(corpus_fn):
        c.extract_ngrams(sentence, words, ngrams)

    assert len(ngrams) == 3
    assert ngrams.word_freq['NMOD:dog-NNS'] == 4
    assert ngrams.word_freq['NMOD:such-NNS NMOD:as-NNS'] == 1

    # Exercise adding probabilities
    ngrams = c.add_ngram_probability(ngrams)

    assert ngrams.ngrams[('NMOD:dog-NNS',)].probability == 0
    assert ngrams.ngrams[('NMOD:dog-NNS', 'NMOD:dog-NNS')].probability == 0
    assert ngrams.ngrams[('NMOD:such-NNS NMOD:as-NNS', 'NMOD:dog-NNS')].probability == 0

    # Save the ngrams
    p = tmpdir.join('ngrams_output.txt')
    c.save_ngrams(ngrams, p)

    contents = p.readlines()
    expected_header = ','.join(['ngram_id', 'ngram', 'lemmas', 'freq', 'probability']) + '\n'

    assert len(contents) == 4
    assert contents[0] == expected_header
    assert contents[1] == '0,NMOD:dog-NNS,dog,1,0\n'
    assert contents[2] == '1,NMOD:dog-NNS NMOD:dog-NNS,dog dog,1,0\n'
    assert contents[3] == '2,NMOD:such-NNS NMOD:as-NNS NMOD:dog-NNS,dog,1,0\n'


def test_save_ngram_stats(tmpdir):
    # Load up test data to populate an frequencies dictionary
    words = c.get_wlist(paths.wlist_fn)
    word_frequencies = {}
    ngrams = c.NgramCollection()

    # Extract the frequencies and ngrams
    for sentence in c.get_sentences(paths.corpus_fn):
        c.extract_frequencies(sentence, words, word_frequencies)
        c.extract_ngrams(sentence, words, ngrams)

    # Save the ngram stats
    p = tmpdir.join('tmp')
    c.save_ngram_stats(ngrams, word_frequencies, p)

    contents = p.readlines()
    expected_header = 'ngram_id,word_id,ngram_index\n'

    assert len(contents) == 129672
    assert contents[0] == expected_header
    assert contents[1] == '0,4291,4\n'
    assert contents[-1] == '42178,1114,12\n'

