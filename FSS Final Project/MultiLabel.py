from nltk.corpus import stopwords, reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from Validation import measureMulti
from Validation import measureNormal
import re
from itertools import count
from scipy.optimize import differential_evolution
import timeit


###Feature Extraction
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

stop_words = stopwords.words("english")

# List of document ids
documents = reuters.fileids()

train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                            documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                           documents))

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

# Tokenisation
#vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
vectorizer = TfidfVectorizer(stop_words=stop_words)
# Learn and transform train documents
vectorised_train_documents = vectorizer.fit_transform(train_docs)
vectorised_test_documents = vectorizer.transform(test_docs)


# Transform multilabel labels
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id)
                                  for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id)
                             for doc_id in test_docs_id])


runtime = []
result = []
predictions = []
#classifiers = [GradientBoostingClassifier()]
classifiers = [LinearSVC(), SGDClassifier(), MultinomialNB(), RandomForestClassifier(),
               LogisticRegression(), DecisionTreeClassifier()]

for clas in classifiers:
    classifier=OneVsRestClassifier(clas)
    start = timeit.default_timer()
    classifier.fit(vectorised_train_documents, train_labels)
    pred = classifier.predict(vectorised_test_documents)
    stop = timeit.default_timer()
    runtime.append(stop-start)
    predictions.append(pred)

#classifier = OneVsRestClassifier(LinearSVC(random_state=42))
#classifier = OneVsRestClassifier(SGDClassifier(alpha=0.001))
#classifier = OneVsRestClassifier(MultinomialNB(alpha=0.01))
#classifier = OneVsRestClassifier(RandomForestClassifier())
#classifier = OneVsRestClassifier(DecisionTreeClassifier())
#classifier = OneVsRestClassifier(LogisticRegression(class_weight='balanced', solver='newton-cg'))
#classifier.fit(vectorised_train_documents, train_labels)


iid = count()
def target(x):
    print(next(iid))
    classifier = OneVsRestClassifier(MultinomialNB(alpha=x[0]))
    classifier.fit(vectorised_train_documents, train_labels)
    predictions = classifier.predict(vectorised_test_documents)
    return -1*measureMulti(test_labels,predictions)

bounds = [(0.00001, 1)]

#result = differential_evolution(target, bounds, maxiter=10, popsize=10, recombination=0.9)
#print("Result: ", result)

for i in range(len(classifiers)):
    print(i)
    r = measureMulti(test_labels,predictions[i])
    result.append(r)

for i in range(len(classifiers)):
    print(print("The time taken by {} was {:.2f}".format(str(classifiers[i]), runtime[i])))
