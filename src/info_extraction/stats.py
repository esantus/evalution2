"""Corpus Statistics Collection

This class opens a corpus in tab-separeted values and it extracts information
about words (a wordlist can be passed to it to reduce the space search). The
output of this class is a tab-separated file (where the fields may contain
comma-separated values), with the following fields:

- Lemma
- Frequency
- Inflection Distribution
- Normalization Distribution
- POS Distribution
- DEP Distribution
- Collocations

Supported languages: EN

"""


