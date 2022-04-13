from ast import literal_eval
# data processing
import numpy as np
import pandas as pd
import seaborn as sns
import dill as pickle
#import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# cleaning data
import re
import nltk
import nltk.tokenize as tok
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
# machine learning models
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


df = pd.read_csv("./companies.csv")
df['tags'] = df['tags'].apply(literal_eval)

tags = []
for i in range(len(df['tags'])):
 tag = [tag.strip().lower().replace("ecommerce","e-commerce") for tag in df['tags'][i]]
 tags.append(tag)

df['tags'] = tags
y_train = df['tags'].values

tags_counts = {}
for tags in y_train:
 for tag in tags:
    if tag in tags_counts:
     tags_counts[tag] += 1
    else:
     tags_counts[tag] = 1


most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:105]

# ToktokTokenizer is a fast, simple, multilingual tokenizer. 
token = tok.ToktokTokenizer()
lemma = WordNetLemmatizer()
punct = '!"$%&\'()*,./:;<=>?@[\\]^_`{|}~'

#print(most_common_tags)
def stopWordsRemove(text):
    stop_words = set(stopwords.words("english"))    
    words = token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    
    return ' '.join(map(str, filtered))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    
    return text

def lemitizeWords(text):
    words = token.tokenize(text)
    listLemma = []
    
    for w in words:
        x = lemma.lemmatize(w, pos = "v")
        listLemma.append(x)
        
    return ' '.join(map(str, listLemma))


def filter_list():
 for i in range(0,len(df)):
   new_tags = []
   for wanted_tag in most_common_tags:
    if wanted_tag[0] in df['tags'][i]:
     if wanted_tag[0] == 'healthcare':
      print("healthcare")
      new_tags.append("health-care")
      print(wanted_tag[0])
     if wanted_tag[0] == 'mobile apps':
      print("mobile apps")
      new_tags.append("apps")
      print(wanted_tag[0])
     if wanted_tag[0] == 'hr':
      print("hr")
      new_tags.append("human-resources")
      print(wanted_tag[0])
     if wanted_tag[0] == 'ai':
      new_tags.append("artificial-intelligence")
      print(wanted_tag[0])
     if wanted_tag[0] == 'hr tech':
      print("hr")
      new_tags.append("human-resources")
      print(wanted_tag[0])
     else:
      new_tags.append(wanted_tag[0].replace(" ","-"))
      print(wanted_tag[0])
     #print(df['tags'][i])
   if len(new_tags) < 1:
     print("SLET ROW")
     df.drop([i], axis=0, inplace=True)
   else:
     df['tags'][i] = new_tags

 #print(df)

 
 
 # Converting html to text in the body
 df['name'] = df['name'].apply(lambda x: clean_text(x)) 
 df['name'] = df['name'].apply(lambda x: lemitizeWords(x)) 
 df['name'] = df['name'].apply(lambda x: stopWordsRemove(x)) 

 # using split() to tokenize each tag using space.
 vectorizer = CountVectorizer(tokenizer = lambda x: x.split())

# fit_transform() does two functions: 
# 1. it fits the model and learns the vocabulary; 
# 2. it transforms training data into feature vectors. The input to fit_transform should be a list of strings.
 
 
 df['tags'] = df['tags'].apply(lambda x: " ".join(x)) 
 tags = vectorizer.fit_transform(df['tags'])


 print("Number of data points :", tags.shape[0])
 print("Number of unique tags :", tags.shape[1])
 # get_feature_name() gives the vocabulary.
 tags_feat = vectorizer.get_feature_names()

 print("Some of the tags found in the dataset:")
 print(tags_feat[:10])

 #Storing the count of tag in each question to list 'tag_count'
 tag_count = tags.sum(axis = 1).tolist()

 # Converting each value in the 'tag_count' to integer.
 tag_count = [int(j) for i in tag_count for j in i]

 print('Total datapoints:', len(tag_count))
 print('Number of tags for the first 10 questions:', tag_count[:10])  

 # count frequency of each tag
 freqs = tags.sum(axis = 0).A1 
 tags_freq = pd.DataFrame(tags_feat, columns = ['tags'])
 tags_freq['Frequency'] = freqs
 tags_freq.head()
 # sort each tag in descending order according ot the number of occurence
 sorted_tags = tags_freq.sort_values(['Frequency'], ascending = False) 
 print(sorted_tags)
 sorted_tags.head()

 

 tfidf_vectorizer = TfidfVectorizer(min_df = 0.00009, max_features = 200000, smooth_idf = True, norm = "l2", 
                                   tokenizer = lambda x: x.split(), sublinear_tf = False, ngram_range = (1, 2))
 count_vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary = 'true', max_features = 500)


 vectorize_post = tfidf_vectorizer.fit_transform(df['name'])
 
 vectorize_tags = count_vectorizer.fit_transform(df['tags'])

 pickle.dump(tfidf_vectorizer, open("vecto", "wb"))

 x_train, x_test, y_train, y_test = train_test_split(vectorize_post, vectorize_tags, random_state = 42, test_size = 0.2, 
                                                    shuffle = False)
 print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


 def train_model(model, x_train, x_test, y_train, y_test):
    model_name = model
    model_name.fit(x_train, y_train)
    y_pred = model_name.predict(x_test)

    f1 = f1_score(y_test, y_pred, average = 'micro')
    precision = precision_score(y_test, y_pred, average = 'micro')
    recall = recall_score(y_test, y_pred, average = 'micro')
    accuracy = accuracy_score(y_test, y_pred)

    # Evaluation Metrics 
    print("Micro F1 Score: %.3f" % (f1))
    print("Micro Precision: %.3f" % (precision))
    print("Micro Recall: %.3f" % (recall))
    print("Accuracy: %.3f" % (accuracy))
    
    pickle.dump(model_name, open("industry", 'wb'))

  
    return f1, precision, recall, accuracy, y_pred

 def max_features_CountVect(n):
    tfidf_vectorizer = TfidfVectorizer(min_df = 0.00009, max_features = 200000, smooth_idf = True, norm = "l2", \
                                       tokenizer = lambda x: x.split(), sublinear_tf = False, ngram_range = (1, 2))
    count_vectorizer_n = CountVectorizer(tokenizer = lambda x: x.split(), binary = 'true', max_features = n)

    vectorize_post = tfidf_vectorizer.fit_transform(df['name'])
    vectorize_tags_n = count_vectorizer_n.fit_transform(df['tags'])


    x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(vectorize_post, vectorize_tags_n, random_state = 42, 
                                                                test_size = 0.2, shuffle = False)

    pickle.dump(count_vectorizer_n, open("count", 'wb'))

    print(x_train_n.shape, x_test_n.shape, y_train_n.shape, y_test_n.shape)    
    return x_train_n, x_test_n, y_train_n, y_test_n, count_vectorizer_n

 x_train_50, x_test_50, y_train_50, y_test_50, count_vectorizer_50 = max_features_CountVect(50)

 dtc = DecisionTreeClassifier()                                                                                                 
 dt_f1_50, dt_precision_50, dt_recall_50, dt_accuracy_50, y_pred_dt50 = train_model(OneVsRestClassifier(dtc, n_jobs = -1), 
                                                                                   x_train_50, x_test_50, 
                                                                                   y_train_50, y_test_50)
 actual_tags = count_vectorizer.inverse_transform(y_test)
 pred_tags_50 = count_vectorizer_50.inverse_transform(y_pred_dt50)  
 
 tags_table = pd.DataFrame({'Actual Tags (y_test)': actual_tags, 'Predicted tags': pred_tags_50})

 print(tags_table.head(50))


 
filter_list()
