"""
This module includes a class for the creation and management of DSMs (either window-
or dep-based), including methods for using the vectors in machine learning
algorithms, for the identification of semantic relations.

Most of the code is inherited from:
"""

import os
import gzip
import pickle
import numpy as np

from collections import defaultdict
from composes.utils import io_utils
from composes.semantic_space.space import Space
from scipy.sparse import coo_matrix, csr_matrix
from composes.matrix.sparse_matrix import SparseMatrix


class DSM(dsm = ""):
    """
    This class allows the user to create and manage a DSM
    """

    def __init__():

        pass



    def combine_vectors(dsm_prefix, combination, word1, word2, WE=False):
        """
        Given two words and a combiantion type, this function returns
        """

        if WE == True:
            A = load_we(word1)
            B = load_we(word2)
        else:
            vector_space = load_pkl_files(dsm_prefix)

            target_index = { w : i for i, w in enumerate(vector_space.id2row) }

            A = load_vector_from_VS(vector_space, word1)
            B = load_vector_from_VS(vector_space, word2)

        if A != False and B != False:
            if combination == "sum":
                return A+B
            elif combination == "mult":
                return A*B
            elif combination == "concat":
                return A.concat(B)
        else:
            return False



    def load_vector(dsm_prefix, word):
        """
        Given a word, it loads and return its vector.

        SIMPLE EXPANSION: RETURN VECTORS FROM PRETRAINED WE WITH OFFSET

        Args:
            dsm_prefix (string): dsm file prefix
            word (string). word vector to retrieve
        Returns:
            The vector or False.
        """

        vector_space = load_pkl_files(dsm_prefix)

        target_index = { w : i for i, w in enumerate(vector_space.id2row) }

        cooc_mat = vector_space.cooccurrence_matrix

        x_index = target_index.get(x, -1)

        if x_index > -1:
            return cooc_mat[x_index, :]
        else:
            return False



    def load_vector_from_VS(vector_space, word):
        """
        Given a word, it loads and return its vector.

        SIMPLE EXPANSION: RETURN VECTORS FROM PRETRAINED WE WITH OFFSET

        Args:
            vector_space (matrix): vector space
            word (string). word vector to retrieve
        Returns:
            The vector or False.
        """

        cooc_mat = vector_space.cooccurrence_matrix

        x_index = target_index.get(x, -1)

        if x_index > -1:
            return cooc_mat[x_index, :]
        else:
            return False



    def load_we(dsm_prefix, word):
        """
        Return the word embedding for a given word.
        """

        if os.path.exists(dsm_prefix+"_offset.txt"):
            with open(...)

                f_we.seek(offset[word])

                return np.array(f_we.readline())



    def create_win_DSM(corpus_directory, frequency_file, output_prefix, win, directional):
        """
        Create window-based co-occurence file from Wackypedia and UKWac

        Args:
            corpus_directory (string): the corpus directory
            frequency_file (string): the file containing lemmas frequencies
            output_prefix (string): the prefix for the output files:
                .sm sparse matrix output file, .rows and .cols
            win (integer): the number of words on each side of the target
            directional (bool): whether (True) or not (False) the contexts should be directional
        Returns:
            Nothing.
        """

        # Load the frequent words file
        with open(frequency_file) as f_in:
            freq_words = set([line.strip() for line in f_in])

            cooc_mat = defaultdict(lambda: defaultdict(int))

            corpus_files = sorted([corpus_dir + '/' + file for file in os.listdir(corpus_directory) if file.endswith('.gz')])

            for file_num, corpus_file in enumerate(corpus_files):

                print('Processing corpus file %s (%d/%d)...' % (corpus_file, file_num + 1, len(corpus_files)))

                for sentence in get_sentences(corpus_file):
                    update_window_based_cooc_matrix(cooc_mat, freq_words, sentence, window_size, directional)

            # Filter contexts
            frequent_contexts = filter_contexts(cooc_mat, MIN_FREQ)

            # Save the files
            save_files(cooc_mat, frequent_contexts, output_prefix)



    def update_window_based_cooc_matrix(cooc_mat, freq_words, sentence, window_size, directional):
        """
        Updates the co-occurrence matrix with the current sentence

        Args:
            cooc_mat (matrix): the co-occurrence matrix
            freq_words (list of strings): the list of frequent words
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

            # Make sure the target is a frequent enough word
            if t_lemma not in freq_words:
                continue

            # print type(t_pos)
            target_lemma = t_lemma + '-' + t_pos[0].decode('utf8').lower()  # lemma + first char of POS, e.g. run-v / run-n

            # Update left contexts if they are inside the window and after BOS (and frequent enough)
            if i > 0:
                for l in range(max(0, i-window_size), i):

                    _, c_lemma, c_pos, _, _, _ = strip_sentence[l]

                    if c_lemma not in freq_words:
                        continue

                    prefix = '-l-' if directional else '-'
                    context = c_lemma + prefix + c_pos[0].decode('utf8').lower()  # context lemma + left + lower pos
                    cooc_mat[target_lemma][context] = cooc_mat[target_lemma][context] + 1

            # Update right contexts if they are inside the window and before EOS (and frequent enough)
            for r in range(i + 1, min(len(strip_sentence), i + window_size + 1)):

                _, c_lemma, c_pos, _, _, _ = strip_sentence[r]

                if c_lemma not in freq_words:
                    continue

                prefix = '-r-' if directional else '-'
                context = c_lemma + prefix + c_pos[0].decode('utf8').lower()
                cooc_mat[target_lemma][context] = cooc_mat[target_lemma][context] + 1

        return cooc_mat


    def n_grams(a, n):
        z = (islice(a, i, None) for i in range(n))
        return zip(*z)




    def create_dep_DSM(corpus_directory, frequency_file, output_prefix):
        """
        This creates a dep-based DSM from a given 6-column-preprocessed corpus.

        Args:
            corpus_directory (string): directory where the corpus is saved
            frequency_file (file): file with frequent words
            ouptut_prefix (string): file prefix for the output
        Returns:
            Nothing.
        """

        # Load the frequent words file
        with open(frequency_file) as f_in:
            freq_words = set([line.strip() for line in f_in])

        cooc_mat = defaultdict(lambda : defaultdict(int))

        corpus_files = sorted([corpus_dir + '/' + file for file in os.listdir(corpus_directory) if file.endswith('.gz')])

        for file_num, corpus_file in enumerate(corpus_files):

            print 'Processing corpus file %s (%d/%d)...' % (corpus_file, file_num + 1, len(corpus_files))

            for sentence in get_sentences(corpus_file):
                update_dep_based_cooc_matrix(cooc_mat, freq_words, sentence)

        # Filter contexts
        frequent_contexts = filter_contexts(cooc_mat, MIN_FREQ)

        # Save the files
        save_files(cooc_mat, frequent_contexts, output_prefix)




    def update_dep_based_cooc_matrix(cooc_mat, freq_words, sentence):
        """
        Updates the co-occurrence matrix with the current sentence

        Args:
            cooc_mat (matrix): the co-occurrence matrix
            freq_words (list of strings): the list of frequent words
            sentence (list of tuples): the current sentence
        Returns:
            The update co-occurrence matrix
        """

        for (word, lemma, pos, index, parent, dep) in sentence:

        # Make sure the target is either a noun, verb or adjective, and it is a frequent enough word
            if lemma not in freq_words or \
                    not (pos.startswith('N') or pos.startswith('V') or pos.startswith('J')):
                continue

            # Not root
            if parent != 0:

                # Get context token and make sure it is either a noun, verb or adjective, and it is
                # a frequent enough word

                # Can't take sentence[parent - 1] because some malformatted tokens might have been skipped!
                parents = [token for token in sentence if token[-2] == parent]

                if len(parents) > 0:
                    _, c_lemma, c_pos, _, _, _ = parents[0]

                    if c_lemma not in freq_words or \
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



    def compute_matrix(dsm_prefix, weight="freq"):
        """
        Given a sparse DSM prefix, it opens the .sm, .rows and .cols and it
        creates the matrix, weighting it according to the desidered weight,
        which is either FREQ, PPMI or PLMI.

        Args:
            dsm_prefix (string): prefix of sparse DSM files (.sm, .rows, .cols)
            weight (string): output weight, which can be freq, ppmi or plmi
        Returns:
            Nothing. It creates a pickle file with the same dsm_prefix
        """

        is_ppmi = True if weight == "ppmi" else False
        is_ppmi = True if weight == "plmi" else False

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
        save_pkl_files(dsm_prefix + postfix, dsm)



    def get_sentences(corpus_file):
        """
        Returns all the (content) sentences in a corpus file.

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



    def save_files(cooc_mat, frequent_contexts, output_prefix):
        """
        Saves the .sm, .rows and .cols files

        Args:
            cooc_mat (matrix): the co-occurrence matrix
            frequent_contexts (wordlist): the frequent contexts
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
                        print >> f_out, ' '.join((target, context, str(freq)))

        # Save the contexts as columns
        with open(output_prefix + '.cols', 'w') as f_out:
            for context in frequent_contexts:
                print >> f_out, context

        # Save the targets as rows
        with open(output_prefix + '.rows', 'w') as f_out:
            for target in cooc_mat.keys():
                print >> f_out, target



    def filter_contexts(cooc_mat, min_occurrences):
        """
        Returns the contexts that occurred at least min occurrences times

        Args:
            cooc_mat (matrix): the co-occurrence matrix
            min_occurrences (integer): the minimum number of occurrences
        Returns:
            The frequent contexts
        """

        context_freq = defaultdict(int)
        for target, contexts in cooc_mat.iteritems():
            for context, freq in contexts.iteritems():
                context_freq[context] = context_freq[context] + freq

        frequent_contexts = set([context for context, frequency in context_freq.iteritems() if frequency >= min_occurrences])
        return frequent_contexts



    def save_pkl_files(dsm_prefix, dsm, save_in_one_file=False):
        """
        Save the space to separate pkl files

        Args:
            dsm_prefix (string): filename prefix
            dsm (matrix): DSM in dense form
        Returns:
            Nothing.
        """

        # Save in a single file (for small spaces)
        if save_in_one_file:
            io_utils.save(dsm, dsm_prefix + '.pkl')

        # Save in multiple files: npz for the matrix and pkl for the other data members of Space
        else:
            mat = coo_matrix(dsm.cooccurrence_matrix.get_mat())
            np.savez_compressed(dsm_prefix + 'cooc.npz', data=mat.data, row=mat.row, col=mat.col, shape=mat.shape)

            with open(dsm_prefix + '_row2id.pkl', 'wb') as f_out:
                pickle.dump(dsm._row2id, f_out, 2)

            with open(dsm_prefix + '_id2row.pkl', 'wb') as f_out:
                pickle.dump(dsm._id2row, f_out, 2)

            with open(dsm_prefix + '_column2id.pkl', 'wb') as f_out:
                pickle.dump(dsm._column2id, f_out, 2)

            with open(dsm_prefix + '_id2column.pkl', 'wb') as f_out:
                pickle.dump(dsm._id2column, f_out, 2)



    def load_pkl_files(dsm_prefix):
        """
        Load the space from either a single pkl file or numerous files

        Args:
            dsm_prefix (string): filename prefix
            dsm (matrix): DSM in dense form
        Returns:
            Nothing.
        """

        # Check whether there is a single pickle file for the Space object
        if os.path.isfile(dsm_prefix + '.pkl'):
            return io_utils.load(dsm_prefix + '.pkl')

        # Load the multiple files: npz for the matrix and pkl for the other data members of Space
        with np.load(dsm_prefix + 'cooc.npz') as loader:
            coo = coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])

        cooccurrence_matrix = SparseMatrix(csr_matrix(coo))

        with open(dsm_prefix + '_row2id.pkl', 'rb') as f_in:
            row2id = pickle.load(f_in)

        with open(dsm_prefix + '_id2row.pkl', 'rb') as f_in:
            id2row = pickle.load(f_in)

        with open(dsm_prefix + '_column2id.pkl', 'rb') as f_in:
            column2id = pickle.load(f_in)

        with open(dsm_prefix + '_id2column.pkl', 'rb') as f_in:
            id2column = pickle.load(f_in)

        return Space(cooccurrence_matrix, id2row, id2column, row2id=row2id, column2id=column2id)
