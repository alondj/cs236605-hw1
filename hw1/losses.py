import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        correct_classes_scores=x_scores[range(x_scores.shape[0]),y]
#         print(f"correct_class_scores\n{correct_classes_scores}")
        loss=x_scores - correct_classes_scores.reshape(-1,1) + self.delta
        loss[range(x_scores.shape[0]),y]-=self.delta
#         print(f"after diff\n{loss}")
        loss = torch.clamp(loss,min=0)
#         print(f"after max\n{loss}")
        

        
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx["x"]=x
        self.grad_ctx["loss_matrix"]=loss
        # ========================

        return loss.sum() / x_scores.shape[0]

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.
        grad = None
        # ====== YOUR CODE: ======
        X_mask = torch.zeros(self.grad_ctx["loss_matrix"].shape)
        X_mask[self.grad_ctx["loss_matrix"] > 0] = 1
        grad = torch.mm(self.grad_ctx["x"].t(),X_mask)
        # ========================

        return grad
