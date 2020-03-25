import abc
import numpy as np


class LossFunction(metaclass=abc.ABCMeta):
    """ Bass class for loss function """
    @abc.abstractmethod
    def init_f_0(self, y):
        """ Initialize F_0 = argmin Sum[L(y_i, c)] """
    @abc.abstractmethod
    def compute_residual(self, f, y):
        """ Calculate residual r = -[dL(y_i, f(x_i))/df(x_i)] """
    @abc.abstractmethod
    def compute_lambda(self, pred, residuals):
        """ Calculate lambda that minimize Sum[L(y_i, F_m-1(x_i) + lambda * h(x_i))] """
    @abc.abstractmethod
    def compute_loss(self, f, y):
        """ Calculate loss function """


class MeanSquareLoss(LossFunction):
    """ MSE Loss: L(y_i, f(x_i)) = (y - f(x_i))^2 """
    def init_f_0(self, y):
        return np.mean(y)

    def compute_residual(self, f, y):
        return 2 * (y - f)

    def compute_lambda(self, pred, residuals):
        A = np.sum(pred * pred)
        B = -2 * np.sum(residuals * pred)
        C = np.sum(residuals * residuals)
        if np.isclose(A, 0) and np.isclose(B, 0):
            return 0.0
        elif np.isclose(A, 0):
            return -C / B
        else:
            return -B / (2 * A)

    def compute_loss(self, f, y):
        return np.mean((y - f)**2)


class LogisticLoss(LossFunction):
    """ Log Loss: L(y_i, f(x_i)) = -(y_i log(f(x_i)) + (1 - y_i) log(1 - f(x_i))) """
    def init_f_0(self, y):
        return np.mean(y)

    def compute_residual(self, f, y):
        return y - f

    def compute_lambda(self, pred, residuals):
        return 1.0

    def compute_loss(self, f, y):
        return 0.5 * (y - f)**2
