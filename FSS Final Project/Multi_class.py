from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
import scipy
import timeit
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn.model_selection import RandomizedSearchCV
from scipy.optimize import differential_evolution
import numpy as np
import warnings

de_iter = 1
rs_iter = 1
def NB(vectorised_train_documents, train_labels, cv):
    model = MultinomialNB()
    param_grid = {'alpha' : scipy.stats.uniform(),
                  'fit_prior' : [True, False],
                 }
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=rs_iter, cv=cv, scoring='f1_micro')
    rsearch.fit(vectorised_train_documents, train_labels)
    return rsearch

def SGD(vectorised_train_documents, train_labels, cv):
    model = SGDClassifier()
    param_grid = {'alpha' : scipy.stats.uniform(),
                  'l1_ratio' : scipy.stats.uniform(),
                  'fit_intercept' : [True, False],
                  'tol': scipy.stats.uniform,
                  'learning_rate' : ['constant', 'optimal', 'invscaling'],
                  'eta0' : scipy.stats.uniform()
                 }
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=rs_iter, cv=cv, scoring='f1_micro')
    rsearch.fit(vectorised_train_documents, train_labels)
    return rsearch

def SVC(vectorised_train_documents, train_labels, cv):
    model = LinearSVC()
    param_grid = {'tol' : scipy.stats.uniform,
                  'C' : scipy.stats.expon(scale=100),
                  'multi_class' : ['ovr', 'crammer_singer'],
                  'fit_intercept' : [True, False],
                  'intercept_scaling' : scipy.stats.uniform
                 }
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=rs_iter, cv=cv, scoring='f1_micro')
    rsearch.fit(vectorised_train_documents, train_labels)
    return rsearch

def RFC(vectorised_train_documents, train_labels, cv):
    model = RandomForestClassifier()
    param_grid = {'n_estimators' : [10,20,30,40,50],
                  'criterion' : ['gini', 'entropy'],
                  'max_features' : ['auto', 'sqrt', 'log2', None],
                  'min_samples_split' : [2,4,6,8],
                  'min_samples_leaf' : [1,2,3,4]
                 }
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=rs_iter, cv=cv, scoring='f1_micro')
    rsearch.fit(vectorised_train_documents, train_labels)
    return rsearch

def LR(vectorised_train_documents, train_labels, cv):
    model = LogisticRegression()
    param_grid = {'tol' : scipy.stats.uniform,
                  'C' : scipy.stats.expon(scale=100),
                  'multi_class' : ['ovr', 'multinomial'],
                  'fit_intercept' : [True, False],
                  'intercept_scaling' : scipy.stats.uniform,
                  'solver' : ['newton-cg', 'lbfgs', 'sag', 'saga']
                 }
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=rs_iter, cv=cv, scoring='f1_micro')
    rsearch.fit(vectorised_train_documents, train_labels)
    return rsearch

def NaiveBayes(vectorised_train_documents, train_labels, best_parameters, cv):
    best_alpha = best_parameters.best_estimator_.alpha
    best_fit_prior = best_parameters.best_estimator_.fit_prior
    best_model = MultinomialNB(alpha=best_alpha, fit_prior=best_fit_prior)
    print(evaluate_cross_validation(best_model, vectorised_train_documents, train_labels, cv))

def SupportVectorClassifier(vectorised_train_documents, train_labels, best_parameters, cv):
    best_tol = best_parameters.best_estimator_.tol
    best_C = best_parameters.best_estimator_.C
    best_multi_class = best_parameters.best_estimator_.multi_class
    best_fit_intercept = best_parameters.best_estimator_.fit_intercept
    best_intercept_scaling = best_parameters.best_estimator_.intercept_scaling
    best_model = LinearSVC(tol=best_tol, C=best_C,multi_class=best_multi_class,fit_intercept=best_fit_intercept,intercept_scaling=best_intercept_scaling)
    print(evaluate_cross_validation(best_model, vectorised_train_documents, train_labels, cv))

def StochasticGradientDescent(vectorised_train_documents, train_labels, best_parameters, cv):
    best_alpha = best_parameters.best_estimator_.alpha
    best_l1_ratio = best_parameters.best_estimator_.l1_ratio
    best_fit_intercept = best_parameters.best_estimator_.fit_intercept
    best_tol = best_parameters.best_estimator_.tol
    best_learning_rate = best_parameters.best_estimator_.learning_rate
    best_eta0 = best_parameters.best_estimator_.eta0
    best_model = SGDClassifier(alpha=best_alpha, l1_ratio=best_l1_ratio, fit_intercept=best_fit_intercept, tol=best_tol, learning_rate=best_learning_rate, eta0=best_eta0)
    print(evaluate_cross_validation(best_model, vectorised_train_documents, train_labels, cv))

def LogisticRegressionClassifier(vectorised_train_documents, train_labels, best_parameters, cv):
    best_tol = best_parameters.best_estimator_.tol
    best_C = best_parameters.best_estimator_.C
    best_multi_class = best_parameters.best_estimator_.multi_class
    best_fit_intercept = best_parameters.best_estimator_.fit_intercept
    best_intercept_scaling = best_parameters.best_estimator_.intercept_scaling
    best_solver = best_parameters.best_estimator_.solver
    best_model = LogisticRegression(tol=best_tol, C=best_C,multi_class=best_multi_class,fit_intercept=best_fit_intercept,intercept_scaling=best_intercept_scaling, solver=best_solver)
    print(evaluate_cross_validation(best_model, vectorised_train_documents, train_labels, cv))

def RandomForestTreesClassifier(vectorised_train_documents, train_labels, vectorised_test_documents, test_labels, best_parameters):
    best_n_estimators = best_parameters.best_estimator_.n_estimators
    best_criterion = best_parameters.best_estimator_.criterion
    best_max_features = best_parameters.best_estimator_.max_features
    best_min_samples_split = best_parameters.best_estimator_.min_samples_split
    best_min_samples_leaf = best_parameters.best_estimator_.min_samples_leaf
    best_model = RandomForestClassifier(n_estimators=best_n_estimators,criterion=best_criterion,max_features=best_max_features,min_samples_split=best_min_samples_split,min_samples_leaf=best_min_samples_leaf)
    print(evaluate_cross_validation(best_model, vectorised_train_documents, train_labels))

def evaluate_cross_validation(clf, X, y, cv):
    scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_micro')
    return scores

def target_NB(x, *args):
    classifier = MultinomialNB(alpha=x[0])
    scores = evaluate_cross_validation(classifier, args[0], args[1], args[2])
    f1 = np.mean(scores)
    return -1*f1

def target_SGD(x, *args):
    classifier = SGDClassifier(alpha=x[0], l1_ratio=x[1], power_t=x[2])
    scores = evaluate_cross_validation(classifier, args[0], args[1], args[2])
    f1 = np.mean(scores)
    return -1*f1

def target_SVC(x, *args):
    classifier = LinearSVC(C=x[0], tol=x[1])
    scores = evaluate_cross_validation(classifier, args[0], args[1], args[2])
    f1 = np.mean(scores)
    return -1*f1

def target_LR(x, *args):
    classifier = LogisticRegression(C=x[0], tol=x[1])
    scores = evaluate_cross_validation(classifier, args[0], args[1], args[2])
    f1 = np.mean(scores)
    return -1*f1

def target_RF(x, *args):
    classifier = RandomForestClassifier(n_estimators=int(x[0]), min_samples_leaf=int(x[1]))
    scores = evaluate_cross_validation(classifier, args[0], args[1], args[2])
    f1 = np.mean(scores)
    return -1*f1

def main():
    warnings.filterwarnings("ignore")
    news = fetch_20newsgroups(subset='all')
    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(news.data)
    y = news.target
    cv = KFold(len(y), 10, shuffle=True, random_state=0)
    #'''
    print('NB without parameter tuning')
    print(evaluate_cross_validation(MultinomialNB(), X, y, cv))
    print('SGD without parameter tuning')
    print(evaluate_cross_validation(SGDClassifier(), X, y, cv))
    print('SVC without parameter tuning')
    print(evaluate_cross_validation(LinearSVC(), X, y, cv))
    print('LR without parameter tuning')
    print(evaluate_cross_validation(LogisticRegression(), X, y, cv))
    #'''
    args = (X, y, cv)
    print("------NB------")
    bounds = [(0.0001, 1)]
    start = timeit.default_timer()
    result = differential_evolution(target_NB, bounds, args=args, maxiter=de_iter)
    end = timeit.default_timer()
    print("DE runtime: ", (end - start))
    start = timeit.default_timer()
    best_model = MultinomialNB(alpha=result.x[0])
    print(evaluate_cross_validation(best_model, X, y, cv))
    end = timeit.default_timer()
    print("NB Runtime: ", (end - start))

    print("------SGD------")
    bounds = [(0.0001, 1), (0,1), (0,1)]
    start = timeit.default_timer()
    result = differential_evolution(target_SGD, bounds, args=args, maxiter=de_iter)
    end = timeit.default_timer()
    print("DE runtime: ", (end - start))
    start = timeit.default_timer()
    best_model = SGDClassifier(alpha=result.x[0], l1_ratio=result.x[1], power_t=result.x[2])
    print(evaluate_cross_validation(best_model, X, y, cv))
    end = timeit.default_timer()
    print("SGD Runtime: ", (end - start))

    print("------SVC------")
    bounds = [(0, 10), (0,1)]
    start = timeit.default_timer()
    result = differential_evolution(target_SVC, bounds, args=args, maxiter=de_iter)
    end = timeit.default_timer()
    print("DE runtime: ", (end - start))
    start = timeit.default_timer()
    best_model = SVC(C=result.x[0], tol=result.x[1])
    print(evaluate_cross_validation(best_model, X, y, cv))
    end = timeit.default_timer()
    print("SVC Runtime: ", (end - start))

    print("------LR------")
    bounds = [(0, 10), (0,1)]
    start = timeit.default_timer()
    result = differential_evolution(target_LR, bounds, args=args, maxiter=de_iter)
    end = timeit.default_timer()
    print("DE runtime: ", (end - start))
    start = timeit.default_timer()
    best_model = LogisticRegression(C=result.x[0], tol=result.x[1])
    print(evaluate_cross_validation(best_model, X, y, cv))
    end = timeit.default_timer()
    print("LR Runtime: ", (end - start))

    print("------RF------")
    bounds = [(5, 100), (1, 10)]
    start = timeit.default_timer()
    result = differential_evolution(target_RF, bounds, args=args, maxiter=de_iter)
    end = timeit.default_timer()
    print("DE runtime: ", (end - start))
    start = timeit.default_timer()
    best_model = RandomForestClassifier(n_estimators=int(result.x[0]), min_samples_leaf=int(result.x[1]))
    print(evaluate_cross_validation(best_model, X, y, cv))
    end = timeit.default_timer()
    print("RF Runtime: ", (end - start))

    #'''
    print("------NB------")
    start = timeit.default_timer()
    best_parameters = NB(X, y, cv)
    end = timeit.default_timer()
    print("Random search runtime: ", (end - start))
    start = timeit.default_timer()
    NaiveBayes(X,y, best_parameters, cv)
    end = timeit.default_timer()
    print("NB Runtime: ", (end - start))
    #'''
    #'''
    print("------SVC------")
    start = timeit.default_timer()
    best_parameters = SVC(X, y, cv)
    end = timeit.default_timer()
    print("Random search runtime: ", (end - start))
    start = timeit.default_timer()
    SupportVectorClassifier(X, y, best_parameters, cv)
    end = timeit.default_timer()
    print("SVC Runtime: ", (end - start))
    #'''
    #'''
    print("------RFC------")
    start = timeit.default_timer()
    best_parameters = RFC(X, y, cv)
    end = timeit.default_timer()
    print("Random search runtime: ", (end - start))
    start = timeit.default_timer()
    RandomForestTreesClassifier(X, y, best_parameters, cv)
    end = timeit.default_timer()
    print("RFC Runtime: ", (end - start))
    #'''
    #'''
    print("------LRC------")
    start = timeit.default_timer()
    best_parameters = LR(X, y, cv)
    end = timeit.default_timer()
    print("Random search runtime: ", (end - start))
    start = timeit.default_timer()
    LogisticRegression(X, y, best_parameters, cv)
    end = timeit.default_timer()
    print("RFC Runtime: ", (end - start))
    #'''
    #'''
    print("------SGD------")
    start = timeit.default_timer()
    best_parameters = SGD(X, y, cv)
    end = timeit.default_timer()
    print("Random search runtime: ", (end - start))
    start = timeit.default_timer()
    StochasticGradientDescent(X, y, best_parameters, cv)
    end = timeit.default_timer()
    print("RFC Runtime: ", (end - start))
    #'''

if __name__ == '__main__':
    main()