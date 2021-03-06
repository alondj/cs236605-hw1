import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.model_selection import KFold

class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # TODO: Calculate the model prediction, y_pred
        
        
        # ====== YOUR CODE: ======
        y_pred = np.matmul(X,self.weights_)
        # ========================
        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)
        # TODO: Calculate the optimal weights using the closed-form solution
        # Use only numpy functions.
        w_opt = None
        # ====== YOUR CODE: ======
        N=1.0*X.shape[0]
        bias_mask=np.identity(X.shape[1])
        bias_mask[0][0]=0
        w_opt=np.linalg.inv(X.T.dot(X)+ N*self.reg_lambda*bias_mask).dot(X.T.dot(y))
        
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: A tensor of shape (N,D) where N is the batch size or of shape
            (D,) (which assumes N=1).
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X)

        # TODO: Add bias term to X as the first feature.

        xb = None
        # ====== YOUR CODE: ======
        xb=np.concatenate((np.ones(X.shape[0],dtype=np.float32).reshape(-1,1),X),axis=1)
        # ========================
      
        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """
    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
#         check_is_fitted(self, ['n_features_', 'n_output_features_'])

        # TODO: Transform the features of X into new features in X_transformed
        # Note: You can count on the order of features in the Boston dataset
        # (this class is "Boston-specific"). For example X[:,1] is the second
        # feature ('ZN').

#         TO ADD
#         1/crime_rate
#         1/lstat
#         rm-5
#         log DIS
#         sqrt(100 -AGE)
        
#         TO REMOVE
#         CHAS
#         crime_rate
#         lstat
#         rm
#         DIS
#         AGE
#         RAD
        
#         X_transformed = np.delete(X,3,1)#remove CHAS
#         X_transformed = np.delete(X_transformed,7,1)#remove CR
#         # ====== YOUR CODE: ======
        X[:,13] =1.0 / X[:,13]#lstat
#         X_transformed[:,0] =1.0 / X_transformed[:,0]#CR
#         X_transformed[:,4] =X_transformed[:,4]-5#RM
#         X_transformed[:,5] =np.log(X_transformed[:,5])#DIS
#         X_transformed[:,6] =np.sqrt(100-X_transformed[:,6])#AGE
#         # ========================

        
        return PolynomialFeatures(self.degree).fit_transform(X)


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    top_n=df.corr()[target_feature].drop([target_feature]).abs().nlargest(n)
    # ========================
    return list(top_n.index), list(top_n.values)


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #
    # Notes:
    # - You can implement it yourself or use the built in sklearn utilities
    #   (recommended). See the docs for the sklearn.model_selection package
    #   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # - If your model has more hyperparameters (not just lambda and degree)
    #   you should add them to the search.
    # - Use get_params() on your model to see what hyperparameters is has
    #   and their names. The parameters dict you return should use the same
    #   names as keys.
    # - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    kf = KFold(k_folds)
    count=0
    best_params ={"bostonfeaturestransformer__degree":2,"linearregressor__reg_lambda":0.1}
    best_loss = np.inf
    for deg in degree_range:
        for lam in lambda_range:
            model.set_params(bostonfeaturestransformer__degree=deg,linearregressor__reg_lambda=lam)
            avg_mse=0.0
            count+=1
            
            for train_idx,test_idx in kf.split(X):
                x_train=X[train_idx]
                y_train=y[train_idx]
                model.fit(x_train, y_train)
                y_pred = model.predict(X[test_idx])
                avg_mse += np.square(y[test_idx] - y_pred).sum() /(2*X.shape[0])
            avg_mse /= k_folds
            if avg_mse < best_loss:
                best_loss = avg_mse
                best_params = {"bostonfeaturestransformer__degree":deg,"linearregressor__reg_lambda":lam}          
    # ========================
    print(count)
    return best_params