import pandas as pd
import re 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk as nlp
import nltk
from sklearn.feature_extraction.txt import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv(r"gender_classifier.csv",encoding="latin1")

data = pd.concat([data.gender,data.description],axis=1)

data.dropna(axis=0,inplace=True)

"""
data.gender = [1 if each == "female" else 0 for each in data.gender]


data["description"] = data["description"].apply(lambda x: re.sub("[^a-zA-Z]"," ", x.lower()))


data["description"] = data["description"].apply(lambda x: word_tokenize(x))




data["description"] = data["description"].apply(lambda description: [word for word in description if not word in set(stopwords.words("english"))])

lemma = nlp.WordNetLemmatizer()


data["description"] = data["description"].apply(lambda description: ' '.join([lemma.lemmatize(word) for word in word_tokenize(description)] if isinstance(description, str) else description))


"""
description_list = []

for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)

#    description = [word for word in description if word not in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()
    description =[lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)


max_features = 500

count_vectorizer = CountVectorizer(max_features= max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transfform(description_list).toarray()

print(max_features,count_vectorizer.get_feature_names())


y = data.iloc[:,0].values
x = sparce_matrix

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=42)

nb = GaussianNB()
nb.fit(x_train, y_train)

y_pred = nb.predict(x_test)

print(nb.score(y_pred.reshape(-1,1), y_test))


































