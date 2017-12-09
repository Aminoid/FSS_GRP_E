from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from Validation import measureNormal, measureMulti
import scipy
import timeit
from nltk.corpus import stopwords, reuters
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
import warnings
from sklearn.cross_validation import cross_val_score, KFold
import numpy as np
from scipy.optimize import differential_evolution

de_iter = 15
rs_iter = 100

def evaluate_cross_validation(clf, X, y, cv):
    scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
    return scores

def NB(vectorised_train_documents, train_labels, cv):
    model = MultinomialNB()
    ovr = OneVsRestClassifier(model)
    param_grid = {"estimator__alpha" : scipy.stats.uniform(),
                  "estimator__fit_prior" : [True, False],
                 }
    rsearch = RandomizedSearchCV(ovr, param_distributions=param_grid, n_iter=rs_iter, cv=cv, scoring='f1_macro')
    rsearch.fit(vectorised_train_documents, train_labels)
    return rsearch

def NaiveBayes(X, y, best_parameters, cv):
    best_alpha = best_parameters.best_params_['estimator__alpha']
    best_fit_prior = best_parameters.best_params_['estimator__fit_prior']
    best_model = MultinomialNB(alpha=best_alpha, fit_prior=best_fit_prior)
    ova = OneVsRestClassifier(best_model)
    print(evaluate_cross_validation(ova, X, y, cv))

def SGD(vectorised_train_documents, train_labels, cv):
    model = SGDClassifier()
    ovr = OneVsRestClassifier(model)
    param_grid = {'estimator__alpha' : scipy.stats.uniform(),
                  'estimator__l1_ratio' : scipy.stats.uniform(),
                  'estimator__fit_intercept' : [True, False],
                  'estimator__tol': scipy.stats.uniform,
                  'estimator__learning_rate' : ['constant', 'optimal', 'invscaling'],
                  'estimator__eta0' : scipy.stats.uniform()
                 }
    rsearch = RandomizedSearchCV(ovr, param_distributions=param_grid, n_iter=rs_iter, cv=cv, scoring='f1_macro')
    rsearch.fit(vectorised_train_documents, train_labels)
    return rsearch

def StochasticGradientDescent(X, y, best_parameters, cv):
    best_alpha = best_parameters.best_params_['estimator__alpha']
    best_l1_ratio = best_parameters.best_params_['estimator__l1_ratio']
    best_fit_intercept = best_parameters.best_params_['estimator__fit_intercept']
    best_tol = best_parameters.best_params_['estimator__tol']
    best_learning_rate = best_parameters.best_params_['estimator__learning_rate']
    best_eta0 = best_parameters.best_params_['estimator__eta0']
    best_model = SGDClassifier(alpha=best_alpha, l1_ratio=best_l1_ratio, fit_intercept=best_fit_intercept, tol=best_tol, learning_rate=best_learning_rate, eta0=best_eta0)
    ova = OneVsRestClassifier(best_model)
    print(evaluate_cross_validation(ova, X, y, cv))

def LR(vectorised_train_documents, train_labels):
    model = LogisticRegression()
    ovr = OneVsRestClassifier(model)
    param_grid = {'estimator__tol' : scipy.stats.uniform,
                  'estimator__C' : scipy.stats.expon(scale=100),
                  'estimator__multi_class' : ['ovr', 'multinomial'],
                  'estimator__fit_intercept' : [True, False],
                  'estimator__intercept_scaling' : scipy.stats.uniform,
                  'estimator__solver' : ['newton-cg', 'lbfgs', 'sag', 'saga']
                 }
    rsearch = RandomizedSearchCV(ovr, param_distributions=param_grid, n_iter=rs_iter)
    rsearch.fit(vectorised_train_documents, train_labels)
    return rsearch

def LogisticRegressionClassifier(X, y, best_parameters, cv):
    best_tol = best_parameters.best_estimator_['estimator__tol']
    best_C = best_parameters.best_estimator_['estimator__C']
    best_multi_class = best_parameters.best_estimator_['estimator__multi_class']
    best_fit_intercept = best_parameters.best_estimator_['estimator__fit_intercept']
    best_intercept_scaling = best_parameters.best_estimator_['estimator__intercept_scaling']
    best_solver = best_parameters.best_estimator_['estimator__solver']
    best_model = LogisticRegression(tol=best_tol, C=best_C,multi_class=best_multi_class,fit_intercept=best_fit_intercept,intercept_scaling=best_intercept_scaling, solver=best_solver)
    ova = OneVsRestClassifier(best_model)
    print(evaluate_cross_validation(ova, X, y, cv))

def SVC(vectorised_train_documents, train_labels, cv):
    model = LinearSVC()
    ovr = OneVsRestClassifier(model)
    param_grid = {'estimator__tol' : scipy.stats.uniform,
                  'estimator__C' : scipy.stats.expon(scale=100),
                  'estimator__multi_class' : ['ovr', 'crammer_singer'],
                  'estimator__fit_intercept' : [True, False],
                  'estimator__intercept_scaling' : scipy.stats.uniform
                 }
    rsearch = RandomizedSearchCV(ovr, param_distributions=param_grid, n_iter=rs_iter, cv=cv, scoring='f1_macro')
    rsearch.fit(vectorised_train_documents, train_labels)
    return rsearch

def SupportVectorClassifier(X, y, best_parameters, cv):
    best_tol = best_parameters.best_params_['estimator__tol']
    best_C = best_parameters.best_params_['estimator__C']
    best_multi_class = best_parameters.best_params_['estimator__multi_class']
    best_fit_intercept = best_parameters.best_params_['estimator__fit_intercept']
    best_intercept_scaling = best_parameters.best_params_['estimator__intercept_scaling']
    best_model = LinearSVC(tol=best_tol, C=best_C,multi_class=best_multi_class,fit_intercept=best_fit_intercept,intercept_scaling=best_intercept_scaling)
    ova = OneVsRestClassifier(best_model)
    print(evaluate_cross_validation(ova, X, y, cv))

def RFC(vectorised_train_documents, train_labels):
    model = RandomForestClassifier()
    ovr = OneVsRestClassifier(model)
    param_grid = {'estimator__n_estimators' : [10,20,30,40,50],
                  'estimator__criterion' : ['gini', 'entropy'],
                  'estimator__max_features' : ['auto', 'sqrt', 'log2', None],
                  'estimator__min_samples_split' : [2,4,6,8],
                  'estimator__min_samples_leaf' : [1,2,3,4]
                 }
    rsearch = RandomizedSearchCV(ovr, param_distributions=param_grid, n_iter=rs_iter)
    rsearch.fit(vectorised_train_documents, train_labels)
    return rsearch

def RandomForestTreesClassifier(X, y, best_parameters, cv):
    best_n_estimators = best_parameters.best_estimator_.n_estimators
    best_criterion = best_parameters.best_estimator_.criterion
    best_max_features = best_parameters.best_estimator_.max_features
    best_min_samples_split = best_parameters.best_estimator_.min_samples_split
    best_min_samples_leaf = best_parameters.best_estimator_.min_samples_leaf
    best_model = RandomForestClassifier(n_estimators=best_n_estimators,criterion=best_criterion,max_features=best_max_features,min_samples_split=best_min_samples_split,min_samples_leaf=best_min_samples_leaf)
    ova = OneVsRestClassifier(best_model)
    print(evaluate_cross_validation(ova, X, y, cv))

def target_NB(x, *args):
    classifier = OneVsRestClassifier(MultinomialNB(alpha=x[0]))
    scores = evaluate_cross_validation(classifier, args[0], args[1], args[2])
    f1 = np.mean(scores)
    return -1*f1

def target_SGD(x, *args):
    classifier = OneVsRestClassifier(SGDClassifier(alpha=x[0], l1_ratio=x[1], power_t=x[2]))
    scores = evaluate_cross_validation(classifier, args[0], args[1], args[2])
    f1 = np.mean(scores)
    return -1*f1

def target_SVC(x, *args):
    classifier = OneVsRestClassifier(LinearSVC(C=x[0], tol=x[1]))
    scores = evaluate_cross_validation(classifier, args[0], args[1], args[2])
    f1 = np.mean(scores)
    return -1*f1

def target_LR(x, *args):
    classifier = OneVsRestClassifier(LogisticRegression(C=x[0], tol=x[1]))
    scores = evaluate_cross_validation(classifier, args[0], args[1], args[2])
    f1 = np.mean(scores)
    return -1*f1

def target_RF(x, *args):
    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=int(x[0]), min_samples_leaf=int(x[1])))
    scores = evaluate_cross_validation(classifier, args[0], args[1], args[2])
    f1 = np.mean(scores)
    return -1*f1

def main():
    warnings.filterwarnings("ignore")
    stop_words = stopwords.words("english")
    documents = reuters.fileids()
    X = [reuters.raw(doc_id) for doc_id in documents]
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(X)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform([reuters.categories(doc_id) for doc_id in documents])
    cv = KFold(len(y), 10, shuffle=True, random_state=0)

    args = (X, y, cv)
    print("------NB------")
    bounds = [(0.0001, 1)]
    start = timeit.default_timer()
    result = differential_evolution(target_NB, bounds, args=args, maxiter=de_iter)
    end = timeit.default_timer()
    print("DE runtime: ", (end - start))
    start = timeit.default_timer()
    best_model = OneVsRestClassifier(MultinomialNB(alpha=result.x[0]))
    print(evaluate_cross_validation(best_model, X, y, cv))
    end = timeit.default_timer()
    print("NB Runtime: ", (end - start))

    print("------SGD------")
    bounds = [(0.0001, 1), (0, 1), (0, 1)]
    start = timeit.default_timer()
    result = differential_evolution(target_SGD, bounds, args=args, maxiter=de_iter)
    end = timeit.default_timer()
    print("DE runtime: ", (end - start))
    start = timeit.default_timer()
    best_model = OneVsRestClassifier(SGDClassifier(alpha=result.x[0], l1_ratio=result.x[1], power_t=result.x[2]))
    print(evaluate_cross_validation(best_model, X, y, cv))
    end = timeit.default_timer()
    print("SGD Runtime: ", (end - start))

    print("------SVC------")
    bounds = [(0, 10), (0, 1)]
    start = timeit.default_timer()
    result = differential_evolution(target_SVC, bounds, args=args, maxiter=de_iter)
    end = timeit.default_timer()
    print("DE runtime: ", (end - start))
    start = timeit.default_timer()
    best_model = OneVsRestClassifier(SVC(C=result.x[0], tol=result.x[1]))
    print(evaluate_cross_validation(best_model, X, y, cv))
    end = timeit.default_timer()
    print("SVC Runtime: ", (end - start))

    print("------LR------")
    bounds = [(0, 10), (0, 1)]
    start = timeit.default_timer()
    result = differential_evolution(target_LR, bounds, args=args, maxiter=de_iter)
    end = timeit.default_timer()
    print("DE runtime: ", (end - start))
    start = timeit.default_timer()
    best_model = OneVsRestClassifier(LogisticRegression(C=result.x[0], tol=result.x[1]))
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
    best_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=int(result.x[0]), min_samples_leaf=int(result.x[1])))
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
    NaiveBayes(X,y,best_parameters,cv)
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
    LogisticRegressionClassifier(X, y, best_parameters, cv)
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
