import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader,sampler

import cs236605.dataloader_utils as dataloader_utils
from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)

        # TODO: Implement k-NN class prediction based on distance matrix.
        # For each training sample we'll look for it's k-nearest neighbors.
        # Then we'll predict the label of that sample to be the majority
        # label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        k_nearest=np.argsort(dist_matrix)[:,list(range(self.k))]
        
        k_classes = np.vectorize(lambda x:self.y_train[x])(k_nearest)

        for i in range(n_test):
#             # TODO:
#             # - Find indices of k-nearest neighbors of test sample i
#             # - Set y_pred[i] to the most common class among them

#             # ====== YOUR CODE: ======
              y_pred[i]= torch.tensor((np.argmax(np.bincount(k_classes[i]))))
#             # ========================

        return y_pred

    def calc_distances(self, x_test: Tensor):
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """

        # TODO: Implement L2-distance calculation as efficiently as possible.
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - No credit will be given for an implementation with two explicit
        #   loops.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops). Hint: Open the expression (a-b)^2.

        
        # ====== YOUR CODE: ======
        l2_dists=(x_test.numpy()**2).sum(axis=1)[:,np.newaxis] + (self.x_train.numpy()**2).sum(axis=1)
        l2_dists-= 2*(x_test.numpy().dot(self.x_train.numpy().T))
        l2_dists = np.sqrt(l2_dists)
                       
        # ========================

        return torch.from_numpy(l2_dists)


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1
    

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    accuracy = None
    # ====== YOUR CODE: ======
    accuracy = (y_pred == y)
    # ========================

    return accuracy.sum().item() / y.shape[0]


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []
    
    indices = [list(range((int(j*(len(ds_train)/(num_folds)))) , min( len(ds_train) , int((j+1) * (len(ds_train) / num_folds)) ) )) for j in range(num_folds)]

    
    flatten = lambda l: [item for sublist in l for item in sublist]        
    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)
        model_acc=[]
        for j in range(num_folds):
            train_indices = indices[0:j] + indices[j+1:]
            train_indices = np.array(flatten(train_indices))
            validate_indices = np.array(indices[j])
            train_sampler=sampler.SubsetRandomSampler(train_indices)
            valid_sampler=sampler.SubsetRandomSampler(validate_indices)

                                                                                                                                                        
            dl_train=DataLoader(ds_train,batch_size=32,sampler=train_sampler)                                                                                                                              
            dl_valid=DataLoader(ds_train,batch_size=32,sampler=valid_sampler)
        
            model.train(dl_train)
            x_valid, y_valid = dataloader_utils.flatten(dl_valid)
            
            y_pred=model.predict(x_valid)
            acc=accuracy(y_valid,y_pred)
            model_acc.append(acc)
        accuracies.append(model_acc)    
        
        

        # TODO: Train model num_folds times with different train/val data.
        # Don't use any third-party libraries.
        # You can use your train/validation splitter from part 1 (even if
        # that means that it's not really k-fold CV since it will be a
        # different split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
