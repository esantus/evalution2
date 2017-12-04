"""
This module includes a class for the creation and management of both
window- and dependency-based DSMs. It includes method for their
training from corpora, as well as method for executing operations on
the vectors and train classifiers.

Part of the code is inherited from:
https://github.com/vered1986/UnsupervisedHypernymy
"""

import collections
import gzip
import logging
import os
import pickle
import sys

import numpy as np
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from evalution.composes.matrix.sparse_matrix import SparseMatrix
from evalution.composes.semantic_space.space import Space
from evalution.composes.transformation.scaling.plmi_weighting import PlmiWeighting
from evalution.composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from evalution.composes.utils import io_utils

logger = logging.getLogger(__name__)


class DSM:
    """This class allows the user to create, load and manage DSMs."""
    def __init__(self, dsm_prefix):
        """Initialize the class by Loading a corpus.

        Args:
            dsm_prefix (string): prefix of the DSM file

        Returns:
            dsm (dsm) if success, False otherwise
        """

        self.is_we = False
        self.dsm = False
        self.we_dsm = False

        if dsm_prefix.endswith(".txt"):
            # Try to load the WE.
            logging.info("Loading WE DSM")
            try:
                self.we_file = open(dsm_prefix, "r")
                self.we_dsm = self.load_we(dsm_prefix)
                self.is_we = True
            except Exception as error:
                raise ValueError("%s: cannot load %s." % (error, dsm_prefix))

        elif dsm_prefix.endswith("ppmi") or dsm_prefix.endswith("plmi") or dsm_prefix.endswith("freq"):
            # Load the DSM.
            self.dsm = self.load(dsm_prefix)
        else:
            # Try to compute the dm from the sm.
            try:
                weight = input(
                    "Computing the matrix. Choose between:\n"
                    "\t1) 'no' = Don't compute it\n"
                    "\t2) ppmi/plmi/freq = Chosen weight\n\n")
                if weight.lower() != "no":
                    self.compute_matrix(dsm_prefix, weight)
            except Exception as error:
                logging.error(error)

        if not self.dsm and not self.we_dsm:
            print("You need to create your DSM, as we cannot load the mentioned files.")
            print("Make sure you have given either the prefix of a DSM or WE_DSM\n")
            choice = input(
                "Insert the path if you want to create the matrix:"
                "\n\t1) Write \"no\" to quit without creating the DSM"
                "\n\t2) Write the path where to find the corpus.gz and the wordlist.txt\n\n> ")
            if choice.lower() != "no":
                print("Trying to create a dep-based DSM out of " + choice + " directory")
                # TODO: wrong args
                self.create_sm_dsm(choice, os.path.join(choice, "wordlist.txt"), "depDSM", dep=True)
        elif self.dsm:
            self.word2index = {w: i for i, w in enumerate(self.dsm.id2row)}
        elif self.we_dsm:
            pass

    @staticmethod
    def load(dsm_prefix):
        """
        Load the space from either a single pkl file or numerous .npz files

        Args:
            dsm_prefix (string): filename prefix
        Returns:
            DSM if success, False otherwise.
        """

        # Check whether there is a single pickle file for the Space object
        if os.path.isfile(dsm_prefix + '.pkl'):
            return io_utils.load(dsm_prefix + '.pkl')
        # otherwise load the multiple files: npz for the matrix and pkl for
        # the other data members of Space
        else:
            try:
                with np.load(dsm_prefix + 'cooc.npz') as loader:
                    coo = scipy.sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])),
                                                  shape=loader['shape'])
                cooccurrence_matrix = SparseMatrix(csr_matrix(coo))
                with open(dsm_prefix + '_row2id.pkl', 'rb') as f_in:
                    row2id = pickle.load(f_in)
                with open(dsm_prefix + '_id2row.pkl', 'rb') as f_in:
                    id2row = pickle.load(f_in)
                with open(dsm_prefix + '_column2id.pkl', 'rb') as f_in:
                    column2id = pickle.load(f_in)
                with open(dsm_prefix + '_id2column.pkl', 'rb') as f_in:
                    id2column = pickle.load(f_in)
            except Exception as error:
                print(error)
                return False
            return Space(cooccurrence_matrix, id2row, id2column, row2id=row2id, column2id=column2id)

    @staticmethod
    def save(dsm_prefix, dsm, save_in_one_file=False):
        """
        Save the space to one or more pkl files

        Args:
            dsm_prefix (string): filename prefix
            dsm (matrix): DSM in dense form
            save_in_one_file (bool): to establish whether saving
                or not in a single file
        Returns:
            True if success, False otherwise.
        """

        # Save in a single file (for small spaces)
        if save_in_one_file:
            io_utils.save(dsm, dsm_prefix + '.pkl')

        # Save in multiple files: npz for the matrix and pkl for the
        # other data members of Space
        else:
            try:
                mat = scipy.sparse.coo_matrix(dsm.cooccurrence_matrix.get_mat())
                np.savez_compressed(dsm_prefix + 'cooc.npz', data=mat.data, row=mat.row,
                                    col=mat.col, shape=mat.shape)
                with open(dsm_prefix + '_row2id.pkl', 'wb') as f_out:
                    pickle.dump(dsm._row2id, f_out, 2)
                with open(dsm_prefix + '_id2row.pkl', 'wb') as f_out:
                    pickle.dump(dsm._id2row, f_out, 2)
                with open(dsm_prefix + '_column2id.pkl', 'wb') as f_out:
                    pickle.dump(dsm._column2id, f_out, 2)
                with open(dsm_prefix + '_id2column.pkl', 'wb') as f_out:
                    pickle.dump(dsm._id2column, f_out, 2)
            except Exception as error:
                print(error)
                return False
        return True

    def save_we_offsets(self, dsm_prefix):
        """Get a pretrained Word Embedding txt dsm in input and save for each vector its offset.

        Args:
            dsm_prefix (string): input file.
        Returns:
            Offsets (dictionary) if success, False otherwise.
        """

        try:
            with open(dsm_prefix, "r") as f_in:
                offsets = {}
                current = 0
                for line in f_in:
                    fields = line.split()
                    offsets[fields[0]] = current
                    current += len(line)
            with open(dsm_prefix + '.offsets.pkl', 'wb') as f_out:
                pickle.dump(offsets, f_out, 2)
            print("Offsets saved.")
        except Exception as error:
            print("save_we_offset(): ", error)
            return False
        return offsets

    def load_we(self, dsm_prefix):
        """Load a pretrained Word Embedding txt corpus and its offsets.

        Args:
            dsm_prefix (string): prefix of the input file
        Returns:
            dsm (dsm) if success, False otherwise
        """

        try:
            if os.path.isfile(dsm_prefix + ".offsets.pkl"):
                print("Loading the offsets")
                we_dsm = pickle.load(open(dsm_prefix + ".offsets.pkl", "r"))
            else:
                print("Creating the offsets")
                we_dsm = self.save_we_offsets(dsm_prefix)
        except Exception as error:
            print("load_we(): ", error)
            return False
        return we_dsm

    def create_sm_dsm(self, corpus_directory, target, output_prefix, dep=True, win=0, directional=False):
        """Create a window- or dependency-based co-occurence DSM from Wackypedia and UKWac.

        It is important to mention that DEP-based does not support MWE, but only SWE.

        Args:
            corpus_directory (string): the corpus directory
            target: the file containing the target lemmas
            output_prefix (string): the prefix for the output files:
                .sm sparse matrix output file, .rows and .cols
            dep (bool): True if dependency-based corpus, False otherwise
            win (integer): the number of words (>0) on each side of the target, if dep is False
            directional (bool): whether (True) or not (False) the contexts should be directional
        """

        min_freq = 100

        if not dep and not win:
            print("You cannot create a DSM that is neither window- nor dep-based")
            return False

        # Load the frequent words file
        with open(target, "r") as f_in:
            target_words = [line.strip() for line in f_in]

        cooc_mat = collections.defaultdict(lambda: collections.defaultdict(int))

        corpus_files = sorted(
            [corpus_directory + '/' + file for file in os.listdir(corpus_directory) if file.endswith('.gz')])

        for file_num, corpus_file in enumerate(corpus_files):
            print('Processing corpus file %s (%d/%d)...' % (corpus_file, file_num + 1, len(corpus_files)))
            for sentence in self.get_sentences(corpus_file):
                if dep:
                    self.update_dep_based_cooc_matrix(cooc_mat, target_words, sentence)
                else:
                    self.update_window_based_cooc_matrix(cooc_mat, target_words, sentence, win, directional)
            frequent_contexts = self.filter_contexts(cooc_mat, min_freq)
            self.save_sm_dsm(cooc_mat, frequent_contexts, output_prefix)

            try:
                weight = input(
                    "Computing the matrix. Choose between:\n"
                    "\t1) 'no' = Don't compute it\n"
                    "\t2) 'ppmi/plmi/freq = Chosen weight\n\n")
                if weight.lower() != "no":
                    self.compute_matrix(dsm_prefix, weight)
            except Exception as error:
                print(error)
                return False
        return True

    def update_window_based_cooc_matrix(self, cooc_mat, target_words, sentence, window_size, directional):
        """Update the co-occurrence matrix with the current sentence

        Args:
            cooc_mat (matrix): the co-occurrence matrix
            target_words (list of strings): the file containing the target lemmas
            sentence (list of tuples): the current sentence
            window_size (integer): the number of words on each side of the target
            directional (bool): whether (True) or not (False) to distinguish between
                contexts before and after the target
        Returns:
            The update co-occurrence matrix.
        """

        # Remove all the non relevant words, keeping only NN, JJ and VB
        strip_sentence = [(w_word, w_lemma, w_pos, w_index, w_parent, w_dep) for
                          (w_word, w_lemma, w_pos, w_index, w_parent, w_dep) in sentence
                          if w_pos.startswith('N') or w_pos.startswith('V') or w_pos.startswith('J')]

        # Add 1 for content words in window_size, either differentiating or not the contexts by direction
        for i, (t_word, t_lemma, t_pos, t_index, t_parent, t_dep) in enumerate(strip_sentence):
            # TODO: Add a control for MWEs
            # mwes = [mwe.split(" ") for mwe in target_words if target_words.split(" ") > 1]
            # if t_lemma in zip(*mwe)[0]:
            # Make sure the target is a frequent enough word
            if t_lemma not in target_words:
                continue

            target_lemma = t_lemma + '-' + t_pos[0].decode(
                'utf8').lower()  # lemma + first char of POS, e.g. run-v / run-n
            # Update left contexts if they are inside the window and after BOS (and frequent enough)
            if i > 0:
                for l in range(max(0, i - window_size), i):
                    c_lemma, c_pos = strip_sentence[0][1:3]
                    if c_lemma not in target_words:
                        continue
                    prefix = '-l-' if directional else '-'
                    context = c_lemma + prefix + c_pos[0].decode('utf8').lower()  # context lemma + left + lower pos
                    cooc_mat[target_lemma][context] = cooc_mat[target_lemma][context] + 1

            # Update right contexts if they are inside the window and before EOS (and frequent enough)
            for r in range(i + 1, min(len(strip_sentence), i + window_size + 1)):
                c_lemma, c_pos = strip_sentence[r][1:3]
                if c_lemma not in target_words:
                    continue

                prefix = '-r-' if directional else '-'
                context = c_lemma + prefix + c_pos[0].decode('utf8').lower()
                cooc_mat[target_lemma][context] = cooc_mat[target_lemma][context] + 1

        return cooc_mat

    def filter_contexts(self, cooc_mat, min_occurrences):
        """Return the contexts that occurred at least min occurrences times

        Args:
            cooc_mat (matrix): the co-occurrence matrix
            min_occurrences (integer): the minimum number of occurrences
        Returns:
            The frequent contexts
        """

        context_freq = collections.defaultdict(int)
        for target, contexts in cooc_mat.iteritems():
            for context, freq in contexts.iteritems():
                context_freq[context] = context_freq[context] + freq

        frequent_contexts = set(
            [context for context, frequency in context_freq.items() if frequency >= min_occurrences])
        return frequent_contexts

    def save_sm_dsm(self, cooc_mat, frequent_contexts, output_prefix):
        """
        Save the .sm, .rows and .cols files

        Args:
            cooc_mat (matrix): the co-occurrence matrix
            frequent_contexts (wordlist): list containing the frequent contexts
            output_prefix (string): the prefix for the output files: .sm sparse
                matrix output file, .rows and .cols
        Returns:
            Nothing. It saves the sparse matrix in output_prefix files.
        """

        # Print in a sparse matrix format
        with open(output_prefix + '.sm', 'w') as f_out:
            for target, contexts in cooc_mat.iteritems():
                for context, freq in contexts.iteritems():
                    if context in frequent_contexts:
                        f_out.writelines(' '.join((target, context, str(freq))))

        # Save the contexts as columns
        with open(output_prefix + '.cols', 'w') as f_out:
            for context in frequent_contexts:
                f_out.writelines(context)

        # Save the targets as rows
        with open(output_prefix + '.rows', 'w') as f_out:
            for target in cooc_mat.keys():
                f_out.writelines(target)

    def update_dep_based_cooc_matrix(self, cooc_mat, target_words, sentence):
        """Update the co-occurrence matrix with the current sentence

        Args:
            cooc_mat (matrix): the co-occurrence matrix
            target_words (list of strings): the file containing the target lemmas
            sentence (list of tuples): the current sentence
        Returns:
            The update co-occurrence matrix
        """

        for (word, lemma, pos, index, parent, dep) in sentence:

            # Make sure the target is either a noun, verb or adjective, and it is a frequent enough word
            if lemma not in target_words or \
                    not (pos.startswith('N') or pos.startswith('V') or pos.startswith('J')):
                continue
            # Not root
            if parent != 0:
                # Get context token and make sure it is either a noun, verb or adjective, and it is
                # a frequent enough word
                # Can't take sentence[parent - 1] because some malformatted tokens might have been skipped!
                parents = [token for token in sentence if token[-2] == parent]

                if len(parents) > 0:
                    c_lemma, c_pos = parents[0][1:3]
                    if c_lemma not in target_words or \
                            not (c_pos.startswith('N') or c_pos.startswith('V') or c_pos.startswith('J')):
                        continue
                    target = lemma + '-' + pos[0].lower()  # lemma + first char of POS, e.g. run-v / run-n
                    context = dep + ':' + c_lemma + '-' + c_pos[0].lower()  # dependency label : parent lemma
                    cooc_mat[target][context] = cooc_mat[target][context] + 1
                    # Add the reversed edge
                    reversed_target = c_lemma + '-' + c_pos[0].lower()
                    reversed_context = dep + '-1:' + lemma + '-' + pos[0].lower()
                    cooc_mat[reversed_target][reversed_context] = cooc_mat[reversed_target][reversed_context] + 1
        return cooc_mat

    def get_sentences(self, corpus_file):
        """Return all the (content) sentences in a corpus file.

        Args:
            corpus_file (filename): the corpus file name
        Returns:
            The next sentence (yield)
        """

        # Read all the sentences in the file
        with gzip.open(corpus_file, 'r') as f_in:
            s = []
            for line in f_in:
                line = line.decode('ISO-8859-2')
                # Ignore start and end of doc
                if '<text' in line or '</text' in line or '<s>' in line:
                    continue
                # End of sentence
                elif '</s>' in line:
                    yield s
                    s = []
                else:
                    try:
                        word, lemma, pos, index, parent, dep = line.split()
                        s.append((word, lemma, pos, int(index), int(parent), dep))
                    # One of the items is a space - ignore this token
                    except:
                        continue

    def compute_matrix(self, dsm_prefix, weight="freq"):
        """Given a sparse DSM prefix, open the .sm, .rows and .cols and
        create the matrix, weighting it according to the desidered weight,
        which is either FREQ, PPMI or PLMI.

        Args:
            dsm_prefix (string): prefix of sparse DSM files (.sm, .rows, .cols)
            weight (string): output weight, which can be freq, ppmi or plmi
        Returns:
            Nothing. It creates a pickle file with the same dsm_prefix
        """

        print("Computing matrix")
        is_ppmi = (True if weight == "ppmi" else False)
        is_plmi = (True if weight == "plmi" else False)

        # Create a space from co-occurrence counts in sparse format
        dsm = Space.build(data=dsm_prefix + '.sm',
                          rows=dsm_prefix + '.rows',
                          cols=dsm_prefix + '.cols',
                          format='sm')

        if is_ppmi:
            # Apply ppmi weighting
            dsm = dsm.apply(PpmiWeighting())
            postfix = "_ppmi"
        elif is_plmi:
            # Apply plmi weighting
            dsm = dsm.apply(PlmiWeighting())
            postfix = "_plmi"
        else:
            postfix = "_freq"

        # Save the Space object in pickle format
        self.save(dsm_prefix + postfix, dsm)

    def get(self, word):
        """Passpartout for the specific get_we and get_vec functions

        Args:
            word (string): the vector to be returned
        Returns:
            It calls the right function, which will return the right vector
        """

        if self.is_we:
            return self.get_we_vec(word)
        else:
            return self.get_vec(word)

    def get_vec(self, word):
        """
        Given a word, it returns its vector if it exists in the DSM, False otherwise.

        Args:
            word (string): vector to be retrieved
        Returns:
            Vector if success, False otherwise.
        """

        cooc_mat = self.dsm.cooccurrence_matrix

        word_index = self.word2index.get(word, -1)

        if word_index > -1:
            return cooc_mat[word_index, :]
        else:
            return np.array([0])

    def get_we_vec(self, word):
        """
        Given a word, it returns its vector if it exists in the WE DSM, False otherwise.

        Args:
            word (string): vector to be retrieved
        Returns:
            Vector if success, False otherwise.
        """

        if word[-2] == "-" and (word[-1] == "n" or word[-1] == "v" or word[-1] == "j"):
            word = word[:-2]

        if word in self.we_dsm:
            self.we_file.seek(self.we_dsm[word])
            return np.array([float(value) for value in self.we_file.readline().split()[1:]])
        else:
            print("Could not find the vector for: ", word)
            return np.array([0])

    def add(self, v1, v2):
        """Return a sum of v1 and v2

        Args:
            v1, v2 (np arrays): vectors
        Returns:
            Vector sum
        """
        return v1 + v2

    def multiply(self, v1, v2):
        """Return a multiplication betwee v1 and v2

        Args:
            v1, v2 (np arrays): vectors
        Returns:
            Vector multiplication
        """
        return np.multiply(v1, v2)

    def concatenate(self, v1, v2):
        """Return a concatenation of v1 and v2

        Args:
            v1, v2 (np arrays): vectors
        Returns:
            Vector concatenation
        """
        return np.concatenate((v1, v2), axis=0)

    def cosine(self, v1, v2):
        """Return vector cosine for v1 and v2

        Args:
            v1, v2 (np arrays): vectors
        Returns:
            Vector cosine
        """
        return dot(v1, v2) / (norm(v1) * norm(v2))


def baseline(dataset, dsm):
    """Train the model, test on validation set and test on the final testset.

    Args:
        dataset: ?
        dsm (we_dsm or dsm): distributional semantic models from where to get the vectors
    Returns:
        True if success, False otherwise
    """

    # clf = SVC()
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    # clf = linear_model.LogisticRegression()
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    vecs_gold_sum = []
    vecs_gold_concat = []
    vecs_gold_mult = []
    # vecs_gold_cos = []

    print("Loading the vectors and applying the function...")

    i = 0
    for w1, gold, w2 in dataset:
        v1 = dsm.get(w1)
        v2 = dsm.get(w2)
        if v1.shape == v2.shape:
            print("W1 = ", w1, " | W2 = ", w2, " | REL = ", gold)
            vecs_gold_sum.append((dsm.add(v1, v2), gold))
            vecs_gold_concat.append((dsm.concatenate(v1, v2), gold))
            vecs_gold_mult.append((dsm.multiply(v1, v2), gold))
            # vecs_gold_cos.append((dsm.cos(v1, v2), gold))
            i += 1

    X_sum, y_sum = [zip(*vecs_gold_sum)[0], zip(*vecs_gold_sum)[1]]
    X_concat, y_concat = [zip(*vecs_gold_concat)[0], zip(*vecs_gold_concat)[1]]
    X_mult, y_mult = [zip(*vecs_gold_mult)[0], zip(*vecs_gold_mult)[1]]
    # X_cos, y_cos = [zip(*vecs_gold_cos)[0], zip(*vecs_gold_cos)[1]]

    X_train_sum, X_test_sum, y_train_sum, y_test_sum = train_test_split(X_sum, y_sum, test_size=0.33, random_state=42)
    X_train_concat, X_test_concat, y_train_concat, y_test_concat = train_test_split(X_concat, y_concat, test_size=0.33,
                                                                                    random_state=42)
    X_train_mult, X_test_mult, y_train_mult, y_test_mult = train_test_split(X_mult, y_mult, test_size=0.33,
                                                                            random_state=42)
    # X_train_cos, X_test_cos, y_train_cos, y_test_cos = train_test_split(X_cos, y_cos, test_size=0.33, random_state=42)

    print("Train: ", len(X_train_sum), " items\nTest: ", len(X_test_sum), " items")

    clf.fit(X_train_sum, y_train_sum)
    print("Sum fit, score is:")
    print(clf.score(X_test_sum, y_test_sum))

    clf.fit(X_train_concat, y_train_concat)
    print("Concatenation fit, score is:")
    print(clf.score(X_test_concat, y_test_concat))

    clf.fit(X_train_mult, y_train_mult)
    print("Multiplication fit, score is:")
    print(clf.score(X_test_mult, y_test_mult))

    # clf.fit(X_train_sum, y_train_sum)
    # print("Sum fit, score is:")
    # print(clf.score(X_test_sum, y_test_sum))


def main():
    dsm = DSM(sys.argv[1])
    dataset = [("abstract", "Antonym", "concrete"),
               ("accident", "Antonym", "plan"),
               ("accident", "Antonym", "purpose"),
               ("accident", "Synonym", "event"),
               ("accident", "IsA", "error"),
               ("accident", "IsA", "happen"),
               ("accident", "IsA", "mistake"),
               ("account", "Synonym", "record"),
               ("account", "Synonym", "report"),
               ("account", "Synonym", "score"),
               ("account", "Synonym", "statement"),
               ("account", "IsA", "pay"),
               ("act", "HasProperty", "fun"),
               ("act", "Antonym", "nothing"),
               ("act", "Antonym", "real"),
               ("act", "Synonym", "action"),
               ("act", "Synonym", "performance"),
               ("act", "IsA", "art"),
               ("act", "IsA", "communication"),
               ("act", "IsA", "drama"),
               ("act", "IsA", "perform"),
               ("act", "IsA", "performance"),
               ("act", "IsA", "pretend"),
               ("act", "PartOf", "performance"),
               ("act", "PartOf", "play"),
               ("action", "HasProperty", "excite"),
               ("action", "Synonym", "battle"),
               ("action", "Synonym", "conflict"),
               ("action", "Synonym", "energy"),
               ("action", "Synonym", "exercise"),
               ("action", "Synonym", "movement"),
               ("action", "Synonym", "performance"),
               ("action", "Synonym", "play"),
               ("action", "Synonym", "plot"),
               ("action", "Synonym", "prosecution"),
               ("action", "IsA", "act"),
               ("action", "IsA", "event"),
               ("action", "IsA", "play"),
               ("action", "IsA", "work"),
               ("active", "Antonym", "quiet"),
               ("active", "Antonym", "slow"),
               ("active", "Synonym", "busy"),
               ("active", "Synonym", "work"),
               ("active", "IsA", "person"),
               ("actor", "Synonym", "actress"),
               ("actor", "IsA", "person"),
               ("actor", "PartOf", "film"),
               ("actress", "Antonym", "actor"),
               ("actress", "IsA", "human"),
               ("actress", "IsA", "person"),
               ("add", "Antonym", "remove"),
               ("add", "Antonym", "take"),
               ("add", "Antonym", "take_away"),
               ("add", "Synonym", "join"),
               ("add", "IsA", "increase"),
               ("address", "PartOf", "letter"),
               ("address", "Entails", "speak"),
               ("admire", "Antonym", "dislike"),
               ("admire", "IsA", "look"),
               ("admire", "IsA", "respect"),
               ("adore", "IsA", "love"),
               ("adult", "Antonym", "child"),
               ("adult", "Synonym", "grow_up"),
               ("adult", "IsA", "animal"),
               ("adult", "IsA", "person"),
               ("advance", "Antonym", "back"),
               ("advance", "Antonym", "fall"),
               ("advance", "Antonym", "late"),
               ("advance", "Synonym", "improve"),
               ("advance", "IsA", "increase"),
               ("advance", "IsA", "move"),
               ("adventure", "HasProperty", "thrill"),
               ("adventure", "Antonym", "boredom"),
               ("adventure", "Antonym", "home"),
               ("adventure", "Antonym", "lazy"),
               ("adventure", "Antonym", "nothing"),
               ("adventure", "Antonym", "safe"),
               ("adventure", "IsA", "action"),
               ("adventure", "IsA", "game"),
               ("aeroplane", "HasProperty", "fly"),
               ("aeroplane", "HasA", "seat"),
               ("affair", "Antonym", "marriage"),
               ("affair", "Antonym", "spouse"),
               ("affection", "HasProperty", "healthy"),
               ("affection", "Antonym", "dislike"),
               ("affection", "Synonym", "love"),
               ("affection", "IsA", "feel"),
               ("affection", "IsA", "like"),
               ("age", "Antonym", "young"),
               ("age", "Synonym", "historic_period"),
               ("age", "IsA", "change"),
               ("age", "IsA", "develop"),
               ("age", "PartOf", "life"),
               ("agreement", "HasProperty", "good"),
               ("agreement", "Synonym", "contract"),
               ("agreement", "Synonym", "deal"),
               ("agreement", "Synonym", "understand"),
               ("agreement", "IsA", "statement"),
               ("aid", "Synonym", "help"),
               ("aid", "Synonym", "support"),
               ("air", "HasProperty", "clear"),
               ("air", "HasProperty", "free"),
               ("air", "Antonym", "earth"),
               ("air", "Antonym", "land"),
               ("air", "Antonym", "vacuum"),
               ("air", "Antonym", "water"),
               ("air", "Synonym", "atmosphere"),
               ("air", "HasA", "oxygen"),
               ("air", "HasA", "pressure"),
               ("air", "HasA", "weight"),
               ("air", "IsA", "gas"),
               ("air", "MadeOf", "gas"),
               ("air", "MadeOf", "molecule"),
               ("air", "MadeOf", "oxygen"),
               ("airplane", "HasProperty", "fun"),
               ("airplane", "HasProperty", "heavy"),
               ("airplane", "HasA", "wheel"),
               ("airplane", "HasA", "wing"),
               ("airplane", "IsA", "machine"),
               ("airplane", "IsA", "transportation"),
               ("aisle", "Antonym", "seat"),
               ("aisle", "IsA", "passageway"),
               ("aisle", "PartOf", "store"),
               ("alabama", "IsA", "place"),
               ("alarm", "HasProperty", "loud"),
               ("album", "Synonym", "disk"),
               ("album", "MadeOf", "vinyl"),
               ("alcohol", "HasProperty", "fun"),
               ("alcohol", "Antonym", "water"),
               ("alcohol", "IsA", "drug"),
               ("alcohol", "MadeOf", "rice"),
               ("alcoholic", "IsA", "human"),
               ("all", "Antonym", "nobody"),
               ("all", "Antonym", "nothing"),
               ("alley", "HasProperty", "dark"),
               ("alley", "Synonym", "aisle"),
               ("alley", "Synonym", "walk"),
               ("alley", "IsA", "street"),
               ("alphabet", "Antonym", "number"),
               ("alto", "IsA", "pitch"),
               ("aluminium", "Antonym", "tin"),
               ("aluminium", "Synonym", "aluminum"),
               ("aluminum", "IsA", "gray"),
               ("aluminum", "IsA", "material"),
               ("america", "HasProperty", "huge"),
               ("america", "HasProperty", "violent"),
               ("america", "HasA", "beach"),
               ("america", "HasA", "flag"),
               ("america", "HasA", "president"),
               ("america", "IsA", "continent"),
               ("america", "IsA", "country"),
               ("america", "IsA", "democracy"),
               ("america", "IsA", "place"),
               ("america", "PartOf", "world"),
               ("anchor", "MadeOf", "iron"),
               ("anchor", "PartOf", "vessel"),
               ("angel", "HasProperty", "good"),
               ("angel", "HasA", "wing"),
               ("angel", "IsA", "person"),
               ("anger", "HasProperty", "unpleasant"),
               ("anger", "Antonym", "calm"),
               ("anger", "Antonym", "happiness"),
               ("anger", "Antonym", "happy"),
               ("anger", "Antonym", "love"),
               ("anger", "Synonym", "enrage"),
               ("anger", "IsA", "feel"),
               ("animal", "HasProperty", "alive"),
               ("animal", "HasProperty", "friendly"),
               ("animal", "HasProperty", "pure"),
               ("animal", "Antonym", "bug"),
               ("animal", "Antonym", "human"),
               ("animal", "Antonym", "mineral"),
               ("animal", "Antonym", "person"),
               ("animal", "Antonym", "plant"),
               ("animal", "Antonym", "vegetable"),
               ("animal", "HasA", "baby"),
               ("animal", "HasA", "body"),
               ("animal", "HasA", "bone"),
               ("animal", "HasA", "emotion"),
               ("animal", "HasA", "eye"),
               ("animal", "HasA", "fur"),
               ("animal", "HasA", "meat"),
               ("animal", "HasA", "muscle"),
               ("animal", "HasA", "pain"),
               ("animal", "IsA", "life"),
               ("animal", "IsA", "organism"),
               ("animal", "PartOf", "nature"),
               ("animate", "Antonym", "stationary"),
               ("annoy", "HasProperty", "bad"),
               ("annoy", "Antonym", "please"),
               ("answer", "Antonym", "problem"),
               ("answer", "Antonym", "question"),
               ("answer", "Antonym", "unknown"),
               ("answer", "IsA", "statement"),
               ("antique", "HasProperty", "old"),
               ("antique", "HasProperty", "valuable"),
               ("antique", "HasProperty", "value"),
               ("antique", "Antonym", "new"),
               ("apartment", "HasProperty", "black"),
               ("apartment", "HasProperty", "red"),
               ("apartment", "Synonym", "flat"),
               ("apartment", "HasA", "door"),
               ("apartment", "HasA", "kitchen"),
               ("apartment", "IsA", "build"),
               ("apartment", "IsA", "home"),
               ("apartment", "IsA", "house"),
               ("appear", "Antonym", "leave"),
               ("appear", "Synonym", "arrive"),
               ("appear", "Synonym", "be"),
               ("appear", "Synonym", "come"),
               ("appear", "Synonym", "look"),
               ("appear", "Synonym", "present"),
               ("appear", "Synonym", "show"),
               ("appear", "IsA", "perform"),
               ("apple", "HasProperty", "alive"),
               ("apple", "HasProperty", "good"),
               ("apple", "HasProperty", "green"),
               ("apple", "HasProperty", "healthy"),
               ("apple", "HasProperty", "red"),
               ("apple", "HasProperty", "small"),
               ("apple", "HasProperty", "sweet"),
               ("apple", "HasProperty", "tasty"),
               ("apple", "HasProperty", "yellow"),
               ("apple", "Antonym", "orange"),
               ("apple", "HasA", "core"),
               ("apple", "HasA", "juice"),
               ("apple", "HasA", "peel"),
               ("apple", "HasA", "seed"),
               ("apple", "HasA", "skin"),
               ("apple", "IsA", "computer"),
               ("apple", "IsA", "tree"),
               ("apple", "PartOf", "core"),
               ("approach", "Synonym", "advance"),
               ("approach", "IsA", "address"),
               ("approach", "IsA", "come"),
               ("approach", "IsA", "movement"),
               ("approach", "Entails", "advance"),
               ("arch", "IsA", "bend"),
               ("arch", "IsA", "build"),
               ("arch", "IsA", "form"),
               ("arch", "IsA", "open"),
               ("arch", "PartOf", "bridge"),
               ("arch", "PartOf", "wall"),
               ("argument", "PartOf", "conclusion"),
               ("arise", "Antonym", "get_down"),
               ("arithmetic", "Synonym", "math"),
               ("arithmetic", "IsA", "math"),
               ("arm", "Antonym", "leg"),
               ("arm", "PartOf", "body"),
               ("arm", "PartOf", "human"),
               ("arm", "PartOf", "person"),
               ("armchair", "HasProperty", "comfortable"),
               ("armchair", "HasA", "back"),
               ("armor", "Antonym", "clothe"),
               ("armor", "Synonym", "plate"),
               ("army", "Antonym", "marine"),
               ("army", "Antonym", "navy"),
               ("army", "IsA", "crowd"),
               ("army", "MadeOf", "soldier"),
               ("army", "PartOf", "war"),
               ("arrange", "IsA", "agree"),
               ("art", "HasProperty", "abstract"),
               ("art", "HasProperty", "beautiful"),
               ("art", "Antonym", "science"),
               ("art", "Antonym", "ugly"),
               ("art", "Synonym", "creation"),
               ("art", "IsA", "creation"),
               ("art", "IsA", "hobby"),
               ("art", "IsA", "play"),
               ("article", "HasA", "news"),
               ("article", "IsA", "piece"),
               ("article", "MadeOf", "information"),
               ("article", "PartOf", "document"),
               ("article", "PartOf", "newspaper"),
               ("artist", "HasProperty", "powerful"),
               ("artist", "IsA", "human"),
               ("artist", "IsA", "person"),
               ("ascend", "Antonym", "descend"),
               ("ascend", "Synonym", "climb"),
               ("ascend", "Synonym", "rise"),
               ("ascend", "IsA", "change"),
               ("asia", "HasProperty", "large"),
               ("asia", "IsA", "place"),
               ("asia", "PartOf", "earth"),
               ("assemble", "Antonym", "break"),
               ("assemble", "Antonym", "destroy"),
               ("assemble", "IsA", "make"),
               ("athlete", "HasProperty", "energetic"),
               ("athlete", "IsA", "human"),
               ("athlete", "IsA", "person"),
               ("atmosphere", "Synonym", "feel"),
               ("atmosphere", "Synonym", "mood"),
               ("atmosphere", "IsA", "air"),
               ("atmosphere", "PartOf", "sky"),
               ("atom", "HasA", "electron"),
               ("atom", "MadeOf", "neutron"),
               ("atom", "MadeOf", "nucleus"),
               ("atom", "MadeOf", "proton"),
               ("atom", "PartOf", "molecule"),
               ("attach", "Antonym", "separate"),
               ("attach", "Synonym", "connect"),
               ("attach", "Synonym", "fasten"),
               ("attach", "Synonym", "tie"),
               ("attach", "IsA", "connect"),
               ("attach", "IsA", "join"),
               ("attach", "IsA", "touch"),
               ("australia", "IsA", "continent"),
               ("australia", "IsA", "island"),
               ("australia", "IsA", "place"),
               ("author", "Synonym", "artist"),
               ("author", "Synonym", "writer"),
               ("author", "IsA", "person"),
               ("author", "IsA", "write"),
               ("author", "PartOf", "book"),
               ("authority", "IsA", "book"),
               ("automobile", "Synonym", "car"),
               ("automobile", "HasA", "radio"),
               ("automobile", "HasA", "trunk"),
               ("automobile", "MadeOf", "steel"),
               ("awaken", "IsA", "change"),
               ("ax", "MadeOf", "metal"),
               ("axe", "HasProperty", "sharp"),
               ("baby", "HasProperty", "amuse"),
               ("baby", "HasProperty", "entertain"),
               ("baby", "HasProperty", "fragile"),
               ("baby", "HasProperty", "fun"),
               ("baby", "HasProperty", "happy"),
               ("baby", "HasProperty", "innocent"),
               ("baby", "HasProperty", "nice"),
               ("baby", "HasProperty", "satisfy"),
               ("baby", "HasProperty", "small"),
               ("baby", "HasProperty", "smile"),
               ("baby", "HasProperty", "ugly"),
               ("baby", "HasProperty", "young"),
               ("baby", "Antonym", "adult"),
               ("baby", "Antonym", "man"),
               ("baby", "Antonym", "old"),
               ("baby", "Synonym", "love"),
               ("baby", "HasA", "hair"),
               ("baby", "IsA", "child"),
               ("baby", "IsA", "girl"),
               ("baby", "IsA", "mammal"),
               ("baby", "IsA", "person"),
               ("baby", "PartOf", "family"),
               ("back", "Antonym", "main"),
               ("back", "Synonym", "rear"),
               ("back", "Synonym", "reverse"),
               ("back", "IsA", "confirm"),
               ("back", "PartOf", "chair"),
               ("back", "PartOf", "human"),
               ("back", "PartOf", "trunk"),
               ("bacon", "Synonym", "ham"),
               ("bacon", "MadeOf", "pig"),
               ("bad", "Antonym", "benevolent"),
               ("bad", "Antonym", "good"),
               ("bad", "Antonym", "right"),
               ("bad", "Antonym", "superior"),
               ("bad", "Antonym", "true"),
               ("bad", "IsA", "evil"),
               ("bad", "IsA", "quality"),
               ("bad", "IsA", "result"),
               ("bag", "HasProperty", "blue"),
               ("bag", "Synonym", "baggage"),
               ("bag", "MadeOf", "fabric"),
               ("bag", "MadeOf", "plastic"),
               ("baggage", "Synonym", "gear"),
               ("balance", "Antonym", "bias"),
               ("balance", "Antonym", "prejudice"),
               ("balance", "Synonym", "account"),
               ("balance", "Synonym", "scale"),
               ("balance", "IsA", "match"),
               ("balance", "IsA", "scale"),
               ("balance", "PartOf", "account"),
               ("ball", "HasProperty", "black"),
               ("ball", "HasProperty", "fun"),
               ("ball", "Synonym", "globe"),
               ("ball", "IsA", "baseball"),
               ("ball", "IsA", "event"),
               ("ball", "IsA", "sport"),
               ("ball", "IsA", "toy"),
               ("ball", "MadeOf", "plastic"),
               ("ball", "PartOf", "human"),
               ("balloon", "HasProperty", "colorful"),
               ("balloon", "HasProperty", "hollow"),
               ("balloon", "HasProperty", "rubber"),
               ("balloon", "HasA", "air"),
               ("balloon", "HasA", "gas"),
               ("balloon", "HasA", "water"),
               ("balloon", "IsA", "rubber"),
               ("balloon", "IsA", "sport"),
               ("balloon", "MadeOf", "rubber"),
               ("banana", "HasProperty", "green"),
               ("banana", "HasProperty", "yellow"),
               ("banana", "HasA", "peel"),
               ("banana", "IsA", "dessert"),
               ("band", "Synonym", "ring"),
               ("banjo", "IsA", "string_instrument"),
               ("banjo", "MadeOf", "wood"),
               ("bank", "HasA", "change"),
               ("bank", "HasA", "coin"),
               ("bank", "IsA", "place"),
               ("bank", "IsA", "work"),
               ("bar", "IsA", "business"),
               ("bar", "IsA", "place"),
               ("bar", "IsA", "support"),
               ("bar", "PartOf", "court"),
               ("bar", "PartOf", "goal"),
               ("barbecue", "IsA", "cook"),
               ("bark", "HasProperty", "rough"),
               ("bark", "Antonym", "bite"),
               ("bark", "IsA", "noise"),
               ("bark", "IsA", "talk"),
               ("bark", "IsA", "tan"),
               ("bark", "PartOf", "branch"),
               ("bark", "PartOf", "tree"),
               ("bark", "PartOf", "trunk"),
               ("base", "HasProperty", "black"),
               ("base", "Antonym", "top"),
               ("base", "Synonym", "bad"),
               ("base", "PartOf", "baseball_diamond"),
               ("base", "PartOf", "box"),
               ("base", "PartOf", "vessel"),
               ("baseball", "HasProperty", "hard"),
               ("baseball", "HasProperty", "nice"),
               ("baseball", "HasA", "stitch"),
               ("baseball", "IsA", "action"),
               ("baseball", "IsA", "game"),
               ("baseball", "IsA", "hobby"),
               ("baseball", "IsA", "sport"),
               ("baseball_diamond", "IsA", "place"),
               ("basement", "Antonym", "attic"),
               ("basement", "IsA", "place"),
               ("basement", "PartOf", "house"),
               ("basket", "IsA", "container"),
               ("basket", "MadeOf", "plastic"),
               ("basketball", "HasProperty", "cool"),
               ("basketball", "HasProperty", "fun"),
               ("basketball", "HasProperty", "orange"),
               ("basketball", "HasProperty", "sport"),
               ("basketball", "IsA", "game"),
               ("basketball", "IsA", "sport"),
               ("basketball", "PartOf", "game"),
               ("bat", "Synonym", "baseball_bat"),
               ("bat", "HasA", "eye"),
               ("bat", "HasA", "wing"),
               ("bat", "IsA", "animal"),
               ("bat", "IsA", "club"),
               ("bat", "IsA", "hit"),
               ("bat", "IsA", "stuff"),
               ("bath", "HasProperty", "relax"),
               ("bath", "Antonym", "shower"),
               ("bath", "HasA", "water"),
               ("bath", "IsA", "place"),
               ("bathroom", "HasProperty", "dark"),
               ("bathroom", "Synonym", "bath"),
               ("bathroom", "HasA", "bath"),
               ("bathroom", "HasA", "ceiling"),
               ("bathroom", "HasA", "plumb"),
               ("bathroom", "HasA", "sink"),
               ("bathroom", "HasA", "toilet"),
               ("bathroom", "IsA", "place"),
               ("bathroom", "PartOf", "dwell"),
               ("bathroom", "PartOf", "house"),
               ("bathtub", "Synonym", "bath"),
               ("bathtub", "IsA", "vessel"),
               ("bathtub", "PartOf", "bathroom"),
               ("battle", "Antonym", "peace"),
               ("battle", "Synonym", "conflict"),
               ("battle", "IsA", "fight"),
               ("battle", "PartOf", "war"),
               ("be", "IsA", "state"),
               ("be_alone", "IsA", "choice"),
               ("beach", "HasProperty", "nice"),
               ("beach", "HasProperty", "white"),
               ("beach", "HasA", "sand"),
               ("beach", "IsA", "place"),
               ("beach", "IsA", "shore"),
               ("beach", "MadeOf", "sand"),
               ("beach", "PartOf", "shore"),
               ("beak", "IsA", "body_part"),
               ("beak", "IsA", "mouth"),
               ("beak", "IsA", "nose"),
               ("beak", "PartOf", "bird"),
               ("beam", "Synonym", "smile"),
               ("beam", "IsA", "light"),
               ("bean", "IsA", "film"),
               ("bear", "Antonym", "lion"),
               ("bear", "Synonym", "birth"),
               ("bear", "HasA", "claw"),
               ("bear", "HasA", "fur"),
               ("beat", "Synonym", "hammer"),
               ("beat", "IsA", "move"),
               ("beat", "Entails", "hit"),
               ("beat", "Entails", "win"),
               ("beautiful", "Antonym", "bad"),
               ("beautiful", "Antonym", "dull"),
               ("beautiful", "Antonym", "nasty"),
               ("beautiful", "Antonym", "plain"),
               ("beautiful", "Antonym", "ugly"),
               ("beauty", "HasProperty", "rare"),
               ("beauty", "Antonym", "ugly"),
               ("beauty", "IsA", "person"),
               ("bed", "HasProperty", "big"),
               ("bed", "HasProperty", "comfortable"),
               ("bed", "HasProperty", "flat"),
               ("bed", "HasProperty", "fun"),
               ("bed", "HasProperty", "soft"),
               ("bed", "HasProperty", "square"),
               ("bed", "Antonym", "bath"),
               ("bed", "Antonym", "chair"),
               ("bed", "Antonym", "sofa"),
               ("bed", "IsA", "device")]
    baseline(dataset, dsm)

if __name__ == '__main__':
    main()
