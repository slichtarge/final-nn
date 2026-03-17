# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = np.dot(W_curr, A_prev) + b_curr #linear transform!

        #apply whichever activation fxn we got
        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        else:
            raise ValueError("Unsupported activation function") #invalid activation fxn
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        # X is (batch, features), but we want (features, batch)
        A_curr = X.T 

        #initialize A0 to A_curr
        cache['A0'] = A_curr

        #ok so for each layer
        for idx, layer in enumerate(self.arch):
            #grab layer_idx and A_prev
            layer_idx = idx + 1
            A_prev = A_curr
            
            #grab weights and biases for that layer
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            
            #do a forward pass!
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, layer['activation'])
            
            #save to cache.
            cache['A' + str(layer_idx)] = A_curr
            cache['Z' + str(layer_idx)] = Z_curr
            
        return A_curr, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        m = A_prev.shape[1]
        
        #do backprop according to whichever activation fxn we got
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        
        #calc derivatives
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)
        
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {}
        
        #get gradient according to whichever backprop we are doing
        if self._loss_func == 'bce':
            dA_prev = self._binary_cross_entropy_backprop(y, y_hat)
        else:
            dA_prev = self._mean_squared_error_backprop(y, y_hat)
        
        #step backwards thru the network!!
        rev_layer_indices = reversed(list(enumerate(self.arch)))

        #for each back layer_idx
        for idx, layer in rev_layer_indices:
            layer_idx = idx + 1
            dA_curr = dA_prev
            
            #snag A_prev, Z_curr, W_curr, and b_curr
            A_prev = cache['A' + str(idx)]
            Z_curr = cache['Z' + str(layer_idx)]
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            
            #pass to single_backprop to get the derivs
            dA_prev, dW_curr, db_curr = self._single_backprop(
                W_curr, b_curr, Z_curr, A_prev, dA_curr, layer['activation']
            )
            
            #update new dW_curr and db_curr
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr
            
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        #grab indices of layers
        arch_indices = enumerate(self.arch)

        #for layer
        for idx, _ in arch_indices:
            layer_idx = idx + 1

            #take a lr-sized step in the direction of dW and dB for W and b.
            self._param_dict['W' + str(layer_idx)] -= self._lr * grad_dict['dW' + str(layer_idx)]
            self._param_dict['b' + str(layer_idx)] -= self._lr * grad_dict['db' + str(layer_idx)]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        train_loss_history = []
        val_loss_history = []
        
        #me being careful abt the sizing, want [1, m] not [m,]
        y_train_proc = y_train.reshape(1, -1) 
        y_val_proc = y_val.reshape(1, -1) 

        for epoch in range(self._epochs):
            
            #shuffle em real quick
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train_proc[:, permutation]

            for i in range(0, X_train.shape[0], self._batch_size):

                #simple mini-batching
                X_batch = X_shuffled[i:i + self._batch_size]
                y_batch = y_shuffled[:, i:i + self._batch_size]
                
                #forward, backprop, update params!
                y_hat_batch, cache = self.forward(X_batch)
                grads = self.backprop(y_batch, y_hat_batch, cache)
                self._update_params(grads)
            
            #epoch metrics
            y_hat_train, _ = self.forward(X_train)
            y_hat_val, _ = self.forward(X_val)
            
            #calculate loss according to whichever function we are using
            if self._loss_func == 'bce':
                train_loss = self._binary_cross_entropy(y_train_proc, y_hat_train)
                val_loss = self._binary_cross_entropy(y_val_proc, y_hat_val)
            else:
                train_loss = self._mean_squared_error(y_train_proc, y_hat_train)
                val_loss = self._mean_squared_error(y_val_proc, y_hat_val)
                
            #save along the way
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            
        #return!
        return train_loss_history, val_loss_history

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        return y_hat.T

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        sig = self._sigmoid(Z)
        return dA * sig * (1 - sig)

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0 #every element less than 0 gets set to zero!
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        #bce = (-1/m)(y * log(yhat) + (1-y)*log(1-yhat))

        prev_0 = 1e-15 #don't want np.log(0)

        #clipping y_hat to be in range [prev_0, 1 - prev_0]
        y_hat = np.clip(y_hat, prev_0, 1 - prev_0)

        term1 = y * np.log(y_hat)
        term2 = (1 - y) * np.log(1 - y_hat)

        loss = -np.mean(term1 + term2)
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        
        prev_0 = 1e-15
        #clipping y_hat to be in range [prev_0, 1 - prev_0]
        y_hat = np.clip(y_hat, prev_0, 1 - prev_0)

        numerator = (y_hat - y)
        denominator = (y_hat * (1 - y_hat))
        return numerator/denominator

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        diff = y - y_hat
        diff_squared = diff**2
        return np.mean(diff_squared)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # -2 * (y - y_hat) / y.shape[1]
        diff = y-y_hat
        m = y.shape[1]
        return -1 * diff/m