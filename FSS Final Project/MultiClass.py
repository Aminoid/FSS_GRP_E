from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.multiclass import OneVsRestClassifier
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from Validation import measureNormal
import re
from itertools import count
from scipy.optimize import differential_evolution
import timeit
import scipy
from sklearn.model_selection import RandomizedSearchCV

cachedStopWords = stopwords.words("english")

def tokenize(text):
  min_length = 3
  words = map(lambda word: word.lower(), word_tokenize(text))
  words = [word for word in words if word not in cachedStopWords]
  tokens = (list(map(lambda token: PorterStemmer().stem(token),
                                   words)))
  p = re.compile('[a-zA-Z]+');
  filtered_tokens = list(filter (lambda token: p.match(token) and
                               len(token) >= min_length,
                               tokens))
  return filtered_tokens

###Feature Extraction
stop_words = stopwords.words("english")
vectorizer = TfidfVectorizer(stop_words=stop_words)

train_docs = fetch_20newsgroups(subset='train')
test_docs = fetch_20newsgroups(subset='test')

vectors = vectorizer.fit_transform(train_docs.data)

vectorised_train_documents = vectorizer.fit_transform(train_docs.data)
vectorised_test_documents = vectorizer.transform(test_docs.data)

# Transform multilabel labels
train_labels = train_docs.target
test_labels = test_docs.target

# Classifier
runtime = []
result = []
predictions = []
#classifiers = [GradientBoostingClassifier()]
classifiers = [LinearSVC(), SGDClassifier(), MultinomialNB(), RandomForestClassifier(),
               LogisticRegression(), DecisionTreeClassifier()]
for clas in classifiers:
    classifier=clas
    start = timeit.default_timer()
    print(start)
    classifier.fit(vectorised_train_documents, train_labels)
    pred = classifier.predict(vectorised_test_documents)
    stop = timeit.default_timer()
    print(stop)
    runtime.append(stop-start)
    predictions.append(pred)
#classifier = LinearSVC(random_state=42)
#classifier = SGDClassifier(alpha=0.001)
#classifier = MultinomialNB(alpha=0.01)
#classifier = RandomForestClassifier()
#classifier = MultinomialNB(alpha=0.01)
#classifier = LogisticRegression(class_weight='balanced', solver='newton-cg')



iid = count()
def target(x):
    print(next(iid))
    classifier = OneVsRestClassifier(MultinomialNB(alpha=x[0]))
    classifier.fit(vectorised_train_documents, train_labels)
    predictions = classifier.predict(vectorised_test_documents)
    return -1*measureNormal(test_labels,predictions)

bounds = [(0.00001, 1)]

#result = differential_evolution(target, bounds, maxiter=10, popsize=10, recombination=0.9)
#print("Result: ", result)

for i in range(len(classifiers)):
    print(i)
    r = measureNormal(test_labels,predictions[i])
    result.append(r)

for i in range(len(classifiers)):
    print(print("The time taken by {} was {:.2f}".format(str(classifiers[i]), runtime[i])))
# for r in result:
#
#     print("Precision: {:.4f}"
#           .format(r.get("precision")))

def NB(vectorised_train_documents, train_labels):
    model = MultinomialNB()
    param_grid = {'alpha' : scipy.stats.uniform(),
                  'fit_prior' : [True, False],
                 }
    rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
    rsearch.fit(vectorised_train_documents, train_labels)
    print(rsearch)
    print(rsearch.best_score_)
    print(rsearch.best_estimator_.alpha)

    #model.fit(vectorised_train_documents, train_labels)
    #pred = model.predict(vectorised_test_documents)

def main():
    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    train_docs = fetch_20newsgroups(subset='train')
    test_docs = fetch_20newsgroups(subset='test')

    vectors = vectorizer.fit_transform(train_docs.data)

    vectorised_train_documents = vectorizer.fit_transform(train_docs.data)
    vectorised_test_documents = vectorizer.transform(test_docs.data)

    # Transform multilabel labels
    train_labels = train_docs.target
    test_labels = test_docs.target

    NB(vectorised_train_documents, train_labels)

if __name__ == '__main__':
    main()