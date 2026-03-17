# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import io
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """

    #separate back into original seqs based on boolean labels
    pos_seqs = [s for s, l in zip(seqs, labels) if l is True]
    neg_seqs = [s for s, l in zip(seqs, labels) if l is False]
    
    num_pos = len(pos_seqs)
    num_neg = len(neg_seqs)

    #downsampling strategy: 
    #I'm choosing the number of positives as our target size to keep the ratio 1:1
    if num_neg > num_pos:
        sampled_neg_seqs = random.sample(neg_seqs, num_pos)
        sampled_pos_seqs = pos_seqs
    else:
        #if by some chance negatives are fewer, we keep all negatives 
        #and sample from positives
        sampled_pos_seqs = random.sample(pos_seqs, num_neg)
        sampled_neg_seqs = neg_seqs

    #combine and create labels
    balanced_seqs = sampled_pos_seqs + sampled_neg_seqs
    balanced_labels = [True] * len(sampled_pos_seqs) + [False] * len(sampled_neg_seqs)

    return balanced_seqs, balanced_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    dict = {"A": [1,0,0,0], "T" : [0,1,0,0], "C" : [0,0,1,0], "G" : [0,0,0,1]}
    one_hots = []
    for seq in seq_arr:
        for char in seq:
            if char.upper() not in dict.keys():                                     ##EDGE: invalid char
                    raise ValueError(f"Invalid character: {char}") 
        list_of_lists = [dict.get(char.upper()) for char in seq]
        one_hot_seq = np.concatenate(list_of_lists).tolist()
        one_hots.append(one_hot_seq)

    return one_hots