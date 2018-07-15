# coding=utf-8
"""API to most common queries to the dataset."""

import collections
import os
import pickle
import sqlite3

import tqdm

from typing import AnyStr


def main():
    db_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../data/dataset/evalution2.db'))
    # use verbose=1 for debugging.
    db = EvaldDB(db_path, verbose=0)
    ## get id from name or name from id. Can be used with lang, name, and rel.
    db.lang_id('en')
    db.lang_name(6)
    ## get all words in a language
    # db.all_words('en')
    ## get all synonyms in a language
    # pickle.dump(db.synonyms('en'), open('syns.p', 'wb'))
    ## return True of two words are synonyms.
    print(db.are_syns('Behaviorism', 'Behaviourism'))
    print(db.are_syns('Behaviorism', 'Bank'))
    ## yield all synsets the argument word appears in.
    # for synset in db.synset_of('Bank'):
    #   print(synset)
    # TODO: db.rel_pairs('isa')
    # TODO: db.are_rel('Auto serviço', 'livello', 'isa')

class EvaldDB:
    """A connection object to an evalution db with some useful queries as methods."""
    def __init__(self, db_name, verbose=1):
        if not os.path.exists(db_name):
            answer = input(db_name + ' does not exist. Do you want to download it (230MB)? [Y/n]')
            raise ValueError(answer)

        self.verbose = verbose
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.result_history_len = 5
        self.result_history = collections.deque(maxlen=self.result_history_len)
        self.relations = dict()

    # TODO: refactor this garbage to use an ORM.
    def lang_id(self, lang_name):
        """Return the lang id from the two character language code."""
        return self.query('select language_id from language where language_value like "%s"' %
                          str(lang_name.lower()))[0][0]

    def lang_name(self, lang_code):
        """Return the lang name from the lang id."""
        return self.query('select language_value from language where language_id = %s' % str(lang_code))[0][0]

    def word_id(self, word_name):
        """Return a word value from it's id(s)."""
        return self.query('select word_id from word where word_value = "%s"' % str(word_name.lower()))[0][0]

    def word_name(self, word_id):
        """Return a word value from it's id(s)."""
        return self.query('select word_value from word where word_id = %s' % str(word_id))[0][0]

    def rel_id(self, rel_name):
        """Return a relation id from a relation name."""
        return self.query('select relationName_id from relationname where relationName_value = "%s"'
                          % rel_name.lower())[0][0]

    def rel_name(self, rel_id):
        """Return a relation name from a relation id."""
        return self.query('select relationName_value from relationname where relationName_id = %s' % rel_id)[0][0]

    def all_words(self, lang: AnyStr) -> set():
        """Returns all words in a language.

        Args:
            lang: the two character identifier of the language (e.g. 'en') or its id.
            See docs/langs.txt for a list of lgs.

        Returns:
            A set containing the words in the dataset.
        """
        try:
            int(lang)
        except ValueError:
            lang = self.lang_id(lang)
        # Select subsqueries seem to be way slower.
        word_ids = "select word_id from allwordsenses where language_id = %s" % str(lang)
        # TODO: return self.query(self.word_values(word_ids))
        return self.query('select word_value from word where word_id in (%s)' % word_ids)

    def rel_pairs(self, rel: AnyStr) -> set():
        """Return a set of pairs of words related by rel. If rel is None, returns a set of all words related by any rel.

        Args:
            rel: the relation to detected. See docs/relations.txt for the list of supported relations.

        Returns:
            A set containing tuples with the two words related by the relation specified.
        """
        try:
            int(rel)
        except ValueError:
            rel = self.rel_id(rel)
        # TODO add support for lang.
        pairs = self.query('select sourcesynset_id, targetsynset_id from synsetrelations where relation_id="%s"' % rel)
        return pairs

    def synonyms(self, lang='en'):
        """Returns a set of tuples with all synonyms in a language. The function may take long time to process!

        This function should only be used if you need to alter the main dataset or you need to create a new pickle
        (e.g. for a new language). A python dictionary with all the synsets in available in /data/synsets/.
        """
        # return every sense with more than one word (i.e. a sense with synonyms).
        syns = dict()
        sense_ids = self.query('select wordsense_id, count(*) as c from allwordsenses where language_id = %s '
                               'group by wordsense_id having c = 23' % self.lang_id(lang))
        for no, sense in enumerate(tqdm.tqdm(sense_ids, mininterval=0.5, total=len(sense_ids))):
            words = set(self.query('select ( select word_value from word where word_id = allwordsenses.word_id ) '
                                   'from allwordsenses where wordsense_id = %s' % sense[0]))
            syns[sense[0]] = words
        return syns

    def which_rels(self, w1: AnyStr, w2: AnyStr) -> set():
        """Returns a set containing the relations from w1 to w2 (order sensitive)."""
        pass

    def are_rel(self, w1: AnyStr, w2: AnyStr, rel) -> bool:
        """Returns True if w1 and w2 are related by a rel, if rel is None, return True if w1 and w2 are related by any rel.
        Return False otherwise."""
        pass

    def are_syns(self, w1, w2, lang='en'):
        """Returns true if two words are synonyms."""
        result = self.query('select count(*) from allwordsenses where (word_id = %s or word_id = %s) '
                            'and language_id = %s group by wordsense_id' % (
                             self.word_id(w1), self.word_id(w2), self.lang_id(lang)))
        return True if result[0][0] > 1 else False

    def synset_of(self, word, lang='en', min_len=2):
        """Returns the synsets where `word` appears."""
        # get all the synsets a word appears in
        synsets = self.query('select wordsense_id from allwordsenses '
                             'where language_id = %s and word_id = %s' % (self.lang_id(lang), self.word_id(word)))
        # then get all the words in each of those synsets.
        for sense_id in synsets:
            synsets = [self.word_name(word_id[0]) for word_id in \
                       self.query('select word_id from allwordsenses where wordsense_id = %s' % sense_id[0])]
            # TODO: is it normal that there are so many singleton synsets?
            if len(synsets) >= min_len:
                yield (sense_id, synsets)

    def query(self, sql: AnyStr) -> AnyStr:
        """Execute an arbitrary query."""
        try:
            self.cursor.execute(sql)
        except sqlite3.OperationalError as e:
            print(e)

        result = self.cursor.fetchall()
        if self.verbose:
            print("entries: %d\n\t%s -> %s" % (len(result), sql, result[:5]))
        self.result_history.append((sql, result))
        return self.result_history[-1][1]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        return exc_type, exc_val, exc_tb


main()
