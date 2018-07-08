# coding=utf-8
"""API to most common queries to the dataset."""

import collections
import os
import sqlite3


from typing import AnyStr


class EvaldDB:
    """A connection object to an evalution db with some useful queries as methods."""
    def __init__(self, db_name='/dataset/evaluation2.db'):
        if not os.path.exists(db_name):
            answer = input(db_name + ' does not exist. Do you want to download it (230MB)? [Y/n]')
            print(answer)
            raise ValueError()

        self.conn = sqlite3.connect('example.db')
        self.cursor = self.conn.cursor()
        self.result_history_len = 5
        self.result_history = collections.deque(maxlen=self.result_history_len)

    def lang_id(self, lang_code):
        """Return the lang id from the two character language code."""
        return self.query('select language_id from language where language_value=?', lang_code)

    def all_words(self, lang: AnyStr) -> set():
        """Returns all words in a language.

        Args:
            lang: the two character identifier of the language (e.g. 'en') or its id.
            See docs/langs.txt for a list of lgs.

        Returns:
            A set containing the words in the dataset.
        """

        if not self.query('select word_id from allwordsenses where language_id=?', lang):
            self.query('select word_id from allwordsenses where language_id=?', self.lang_id(lang))
            return self.result_history[-1](1)

    def all_rels(self, rel: AnyStr) -> set():
        """Return a set of pairs of words related by rel. If rel is None, returns a set of all words related by any rel.

        Args:
            rel: the relation to detected. See docs/relations.txt for the list of supported relations.

        Returns:
            A set containing tuples with the two words related by the relation specified.
        """
        return rel

    def which_rels(self, w1: AnyStr, w2: AnyStr) -> set():
        """Returns a set containing the relations on w1 and w2 (order sensitive)."""
        return w1, w2

    def are_rel(self, w1: AnyStr, w2: AnyStr, rel: AnyStr) -> bool:
        """Returns True if w1 and w2 are related by a rel, if rel is None, return True if w1 and w2 are related by any rel.
        Return False otherwise."""
        return True

    def query(self, sql, arg=''):
        """Execute an arbitrary query."""
        self.cursor.execute(sql, arg)
        self.result_history.append(sql, self.cursor.fetchone())
        return self.result_history[-1](1)

    def rows(self):
        return self.cursor.rowcount

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        return exc_type, exc_val, exc_tb
