import numpy as np
import os
import h5py
from tqdm import tqdm
# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.random.uniform(low=-0.02, high=0.02, size=[len(vocab), dim]).astype('float32')
    found = 0
    with open(glove_filename, encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                found += 1
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    print("Found %d out of %d (%.3f)" % (found, len(vocab), found / len(vocab)))
    f = h5py.File("%s.hdf5" % trimmed_filename, "w")
    f.create_dataset("embeddings", shape=embeddings.shape, dtype=np.float32, data=embeddings)
    f.close()
    np.savez_compressed("%s.npyc" % trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks



def flag_bio_tags(gold, predicted, sequence_length=None):
    """Flags chunk matches for the BIO tagging scheme.
    This function will produce the gold flags and the predicted flags. For each aligned
    gold flag ``g`` and predicted flag ``p``:
    * when ``g == p == True``, the chunk has been correctly identified (true positive).
    * when ``g == False and p == True``, the chunk has been incorrectly identified (false positive).
    * when ``g == True and p == False``, the chunk has been missed (false negative).
    * when ``g == p == False``, the chunk has been correctly ignored (true negative).
    Args:
    gold: The gold tags as a Numpy 2D string array.
    predicted: The predicted tags as a Numpy 2D string array.
    sequence_length: The length of each sequence as Numpy array.
    Returns:
    A tuple ``(gold_flags, predicted_flags)``.
    """
    gold_flags = []
    predicted_flags = []

    def _add_true_positive():
        gold_flags.append(True)
        predicted_flags.append(True)

    def _add_false_positive():
        gold_flags.append(False)
        predicted_flags.append(True)

    def _add_true_negative():
        gold_flags.append(False)
        predicted_flags.append(False)

    def _add_false_negative():
        gold_flags.append(True)
        predicted_flags.append(False)

    def _match(ref, hyp, index, length):
        if ref[index] == 1:
            match = True
            while index < length and not ref[index] == 0:
                if ref[index] != hyp[index]:
                    match = False
                index += 1
            match = match and index < length and ref[index] == hyp[index]
            return match, index

        return ref[index] == hyp[index], index

    for b in range(gold.shape[0]):
        length = sequence_length[b] if sequence_length is not None else gold.shape[1]

    # First pass to detect true positives and true/false negatives.
    index = 0
    while index < length:
        gold_tag = gold[b][index]
        match, index = _match(gold[b], predicted[b], index, length)
        if match:
            if gold_tag == 0:
                _add_true_negative()
            else:
                _add_true_positive()
        else:
            if gold_tag != 0:
                _add_false_negative()
        index += 1

    # Second pass to detect false postives.
    index = 0
    while index < length:
        pred_tag = predicted[b][index]
        match, index = _match(predicted[b], gold[b], index, length)
        if not match and pred_tag != 0:
            _add_false_positive()
        index += 1

    return np.array(gold_flags), np.array(predicted_flags)