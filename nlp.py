import numpy as np
from nltk import ngrams
from nltk.collocations import *
import matplotlib.pyplot as plt
import json
import re
import nltk
from nltk.corpus import stopwords
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

stopwords=set(stopwords.words('Turkish'))
words=[]
list_str=[]
list_last=[]
liststopwords=[stopwords]
n=int(input("Select the ngram degree:"))
filecounter=0

while filecounter<10:
    filecounter = filecounter + 1
    f = open(str(filecounter) + '.json', encoding="utf-8")
    data = json.load(f)
    for i in data.values():
        n_gramforpunctuation = re.sub(r'[^\w\s]', "", i)
        n_grams = ngrams(n_gramforpunctuation.split(), 1)
        for grams in n_grams:
            words.append(grams)
print(len(words))
for i in range(len(words)):
    str1 = ''.join(words[i])
    str1_lower= str1.lower()
    list_str.append(str1_lower)

for b in list_str:
    if b not in stopwords:
        list_last.append(b)

def student_t():
    finder.nbest(ngram_measures.student_t, 10)
    b=finder.score_ngrams(ngram_measures.student_t)
    return b

def frequency():
    finder.nbest(ngram_measures.raw_freq, 10)
    results=finder.score_ngrams(ngram_measures.raw_freq)
    return results

def likelihood_ratio():
    finder.nbest(ngram_measures.likelihood_ratio, 10)
    results=finder.score_ngrams(ngram_measures.likelihood_ratio)
    return results

def chisquare2():
    finder.nbest(ngram_measures.chi_sq,10)
    results=finder.score_ngrams(ngram_measures.chi_sq)
    return results

def pmi():
    finder.nbest(ngram_measures.pmi,10)
    results=finder.score_ngrams(ngram_measures.pmi)
    return results

def plotting(variable,title):
    list1=[]
    list2=[]
    for i in range(len(variable)):
        list1.append(variable[i][1])
        list2.append(variable[i][0])
    print(len(variable))

    liste3 =[]

    for c in range(len(variable)):
        x = str((list2[c]))
        x2 = x.replace('(', '').replace(')', '').replace(",", '')
        liste3.append(x2)

    y = list1
    x = liste3

    print(liste3)

    plt.title(title)

    plt.bar(x,y,color ='blue')
    plt.xticks(rotation=90)
    plt.show()

def plotfunc():

    student = student_t()
    freq = frequency()
    like = likelihood_ratio()
    chi = chisquare2()
    pmi_ = pmi()
    plotting(student,"Student_Test "+str(n)+"-grams")
    plotting(freq,"Frequency "+str(n)+"-grams")
    plotting(like,"Likelihood_Ratio "+str(n)+"-grams")
    plotting(chi,"Chi_Square "+str(n)+"-grams")
    plotting(pmi_, "Pointwise Mutual Information " + str(n) + "-grams")

if n == 2:

    ngram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(list_last)
    #finder.apply_freq_filter(3) apply to large file
    plotfunc()

elif n == 3:

    ngram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(list_last)
    finder.apply_freq_filter(3)
    plotfunc()

elif n == 4:

    ngram_measures = nltk.collocations.QuadgramAssocMeasures()
    finder = QuadgramCollocationFinder.from_words(list_last)
    #finder.apply_freq_filter(3) apply to large file
    plotfunc()

elif n==5:

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(list_last)
    vectorizer.get_feature_names_out()
    print(X.shape)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(list_last)
    vectorizer.get_feature_names_out()

    print(X.toarray())

    vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1,1))
    X2 = vectorizer2.fit_transform(list_last)
    vectorizer2.get_feature_names_out()

    print(X2.toarray())

    vocabulary = ['yaralama', 'suç', 'ceza', 'tehdit', 'cana kastetme suçu', 'uyuşturucu madde suçu',
              'trafik suçu', 'izinsiz mal alma suçu','kişiyi hürriyetinden yoksun kılma']

    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                     ('tfid', TfidfTransformer())]).fit(list_last)
    pipe['count'].transform(list_last).toarray()

    pipe1 = pipe['tfid'].idf_
    print(pipe1)
    plt.bar(vocabulary,pipe1)
    plt.show()

    pipe2 = pipe.transform(list_last).shape
    print(pipe2)
elif n==6:

    reg = linear_model.Ridge(alpha=.5)
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    print(reg.coef_)
    print(reg.intercept_)
    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

    print(reg.alpha_),

else:
    print("You can write only 2,3 or 4 grams")
    exit()