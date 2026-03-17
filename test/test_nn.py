# TODO: import dependencies and write unit tests below
import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../nn/')))
from nn import NeuralNetwork 
import preprocess

@pytest.fixture
def simple_nn():
    arch = [
        {'input_dim': 4, 'output_dim': 8, 'activation': 'relu'},
        {'input_dim': 8, 'output_dim': 1, 'activation': 'sigmoid'}
    ]
    return NeuralNetwork(
        nn_arch=arch,
        lr=0.01,
        seed=42,
        batch_size=2,
        epochs=1,
        loss_function='bce'
    )

def test_single_forward(simple_nn):
    W = np.ones((3, 4))
    b = np.zeros((3, 1))
    A_prev = np.ones((4, 2))
    
    #linear part: Z = W.dot(A_prev) + b -> (3,4) @ (4,2) = (3,2)
    #since all are 1s, each element in Z should be 4.0
    A_curr, Z_curr = simple_nn._single_forward(W, b, A_prev, 'relu')
    
    assert Z_curr.shape == (3, 2)
    assert np.allclose(Z_curr, 4.0)
    assert np.allclose(A_curr, 4.0) # ReLU of 4 is 4


def test_forward(simple_nn):
    X = np.random.randn(2, 4) #setting the batch_size=2, features=4
    output, cache = simple_nn.forward(X)
    
    assert output.shape == (1, 2) #the final layer output_dim is 1
    assert 'A0' in cache
    assert 'A1' in cache
    assert 'Z2' in cache
    assert cache['A0'].shape == (4, 2) #transposed input

def test_single_backprop(simple_nn):
    W = np.random.randn(3, 4)
    b = np.random.randn(3, 1)
    Z = np.random.randn(3, 2)
    A_prev = np.random.randn(4, 2)
    dA_curr = np.random.randn(3, 2)
    
    dA_prev, dW_curr, db_curr = simple_nn._single_backprop(
        W, b, Z, A_prev, dA_curr, 'relu'
    )
    
    assert dA_prev.shape == (4, 2)
    assert dW_curr.shape == (3, 4)
    assert db_curr.shape == (3, 1)

def test_predict(simple_nn):
    X = np.random.randn(5, 4)
    predictions = simple_nn.predict(X)
    
    #predict transposes the output back to (samples, output_dim)
    assert predictions.shape == (5, 1)

def test_binary_cross_entropy(simple_nn):
    y = np.array([[1, 0]])
    y_hat = np.array([[0.9, 0.1]])
    loss = simple_nn._binary_cross_entropy(y, y_hat)
    
    assert isinstance(loss, float)
    assert loss > 0

def test_binary_cross_entropy_backprop(simple_nn):
    y = np.array([[1, 0]])
    y_hat = np.array([[0.8, 0.2]])
    dA = simple_nn._binary_cross_entropy_backprop(y, y_hat)
    
    assert dA.shape == (1, 2)

def test_mean_squared_error(simple_nn):
    y = np.array([[1, 2]])
    y_hat = np.array([[1, 4]])
    loss = simple_nn._mean_squared_error(y, y_hat)
    
    #mean of (0^2 + 2^2) = 4/2 = 2.0
    assert np.isclose(loss, 2.0) #answer should be 2

def test_mean_squared_error_backprop(simple_nn):
    y = np.array([[1, 1]])
    y_hat = np.array([[2, 2]])
    dA = simple_nn._mean_squared_error_backprop(y, y_hat)
    
    # -1 * (y - y_hat) / m -> -1 * (-1) / 2 = 0.5
    assert dA.shape == (1, 2)
    assert np.allclose(dA, 0.5) #answer should be 0.5

def test_sample_seqs():
    fake_seqs = ["ATGC", "CCGG", "AAAA", "TTTT", "GGGG"]
    labels = [True, False, False, False, False] # 1 Pos, 4 Neg
    
    sampled_seqs, sampled_labels = preprocess.sample_seqs(fake_seqs, labels)
    
    #make sure they're equal length
    assert len(sampled_seqs) == len(sampled_labels)
    #make sure that we have a 1:1 ratio of positive to negative
    assert sum(sampled_labels) == 1
    assert len(sampled_labels) == 2
    assert sampled_labels.count(True) == sampled_labels.count(False)

def test_one_hot_encode_seqs():

    #do we get an expected result?
    seqs = ["ATC"]
    expected = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]]
    result = preprocess.one_hot_encode_seqs(seqs)
    assert result == expected
    assert len(result[0]) == 4 * len(seqs[0])

    #does an invalid character throw the right error?
    seqs = ["ATNX"]
    with pytest.raises(ValueError, match="Invalid character: N"):
        preprocess.one_hot_encode_seqs(seqs)

    #test uppercase v lowercase
    seqs_upper = ["ATGC"]
    seqs_lower = ["atgc"]
    assert preprocess.one_hot_encode_seqs(seqs_upper) == preprocess.one_hot_encode_seqs(seqs_lower)

    #if we pass multiple seqs do we get the expected dimensions?
    seqs = ["AAA", "TTT", "GGG"]
    result = preprocess.one_hot_encode_seqs(seqs)
    
    assert len(result) == 3 #for the three seqs
    for encoding in result:
        assert len(encoding) == 12 #3 nucleotides * 4 bits
