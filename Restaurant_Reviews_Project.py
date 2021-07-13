import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import re

data = pd.read_csv('s.tsv',delimiter='\t')

X = data['Review']

clean_reviews = []
for i in range(0,len(X)):
    temp = re.sub('[^a-zA-Z]',' ',data['Review'][i])
    temp = temp.lower()
    temp = temp.split()
    ps = PorterStemmer()
    temp = [ps.stem(word) for word in temp if not word in set(stopwords.words('english')) ]
    temp = ' '.join(temp)
    clean_reviews.append(temp)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 800)
#cv = CountVectorizer()
X = cv.fit_transform(clean_reviews)
X = X.toarray()

Y = data['Liked']
Y = Y.values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train,Y_train)

lr.score(X_train,Y_train)
print(lr.score(X_test,Y_test))

