import numpy as np
import random
from frameworks.CPLELearning import CPLELearningModel
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import sklearn.svm
from methods.scikitWQDA import WQDA
from frameworks.SelfLearning import SelfLearningModel

# load data
cancer = load_breast_cancer()
X = cancer.data
ytrue = np.copy(cancer.target).flatten()
ytrue[ytrue > 0] = 1

# label a few points
labeled_N = 4
ys = np.array([-1] * len(ytrue))  # -1 denotes unlabeled point
random_labeled_points = random.sample(
    list(
        np.where(
            ytrue == 0)[0]),
    labeled_N // 2) + random.sample(
    list(
        np.where(
            ytrue == 1)[0]),
    labeled_N // 2)
ys[random_labeled_points] = ytrue[random_labeled_points]

# supervised score
# basemodel = WQDA() # weighted Quadratic Discriminant Analysis
# scikit logistic regression
basemodel = SGDClassifier(loss='log', penalty='l1')
basemodel.fit(X[random_labeled_points, :], ys[random_labeled_points])
print("supervised log.reg. score", basemodel.score(X, ytrue))

# fast (but naive, unsafe) self learning framework
ssmodel = SelfLearningModel(basemodel)
ssmodel.fit(X, ys)
print("self-learning log.reg. score", ssmodel.score(X, ytrue))

# semi-supervised score (base model has to be able to take weighted samples)
ssmodel = CPLELearningModel(basemodel)
ssmodel.fit(X, ys)
print("CPLE semi-supervised log.reg. score", ssmodel.score(X, ytrue))

# semi-supervised score, WQDA model
# weighted Quadratic Discriminant Analysis
ssmodel = CPLELearningModel(WQDA(), predict_from_probabilities=True)
ssmodel.fit(X, ys)
print("CPLE semi-supervised WQDA score", ssmodel.score(X, ytrue))

# semi-supervised score, RBF SVM model
ssmodel = CPLELearningModel(
    sklearn.svm.SVC(
        kernel="rbf",
        probability=True),
    predict_from_probabilities=True)  # RBF SVM
ssmodel.fit(X, ys)
print("CPLE semi-supervised RBF SVM score", ssmodel.score(X, ytrue))
