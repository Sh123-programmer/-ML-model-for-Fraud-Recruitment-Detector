import re
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import ADASYN
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint 
df=pd.read_csv("fake_job_postings.csv")
bool_col=['telecommuting','has_company_logo','has_questions']
for column in df.columns:
    df[column]=df[column].fillna(f'missing_{column}')
for column in bool_col:
    df.loc[df[column]==0,column]=f'no_{column}'
    df.loc[df[column]==1,column]=f'yes_{column}'
cat_col=['employment_type','required_experience','required_education']
for column in cat_col:
    val=list(df[column].values)
    #split join by "_"
    val = ["_".join(re.findall(r"[\w']+",i))+f"_{column}" for i in val]
    df[column]=np.array(val)

train_col=['title', 'location', 'department', 'salary_range',
       'company_profile', 'description', 'requirements', 'benefits',
       'telecommuting', 'has_company_logo', 'has_questions', 'employment_type',
       'required_experience', 'required_education', 'industry', 'function']

initiate=pd.Series(np.full(shape=(df.shape[0],),fill_value=''))
for column in train_col:
    initiate+=(" " + df[column])
#print(initiate.head())
def regexp(text):
    text = re.sub(r'[^A-z0-9\s]','', text) 
    text = re.sub(r'\ss\s','',text)
    text = re.sub(r'  ',' ',text)
    return text
def token(x):
    return nltk.word_tokenize(x)
def lower(text):
    return text.lower()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
lemmatizer=WordNetLemmatizer()
normalized_corpus = np.empty(initiate.shape,dtype='str')
w=[]
for k,text in enumerate(initiate.values):
    filter_sentence=''
    text=lower(text)
    text=regexp(text)
    words=token(text)
    words=[w for w in words if not w in stopword_list]
    for word in words:
        filter_sentence=filter_sentence+' '+str(lemmatizer.lemmatize(word))
    text=filter_sentence
    w.append(text)
vectorizer=TfidfVectorizer(ngram_range=(1,2))
vectorizer.fit(w)
X_train_tfidf = vectorizer.transform(w)
sss=StratifiedShuffleSplit(n_splits=1, random_state=0, test_size=0.2)
X=np.zeros(shape=X_train_tfidf.shape[0],dtype=np.bool_)
y=np.array(df['fraudulent']) #prediction target
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X_train_tfidf[train_index,:], X_train_tfidf[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
adasyn=ADASYN()
X_res, y_res = adasyn.fit_resample(X_train, y_train)
#Feature Selection
selector = SelectFromModel(estimator=LinearSVC()).fit(X_res, y_res)
X_select = selector.transform(X_res)
X_select_test = selector.transform(X_test)
#classifier
rf=RandomForestClassifier().fit(X_select,y_res)
print(f"Random Forest train accuracy: {accuracy_score(rf.predict(X_select),y_res)}")
print(f"Random Forest test accuracy: {accuracy_score(rf.predict(X_select_test),y_test)}")
svc = LinearSVC().fit(X_select, y_res)
print(f"SVC train accuracy: {accuracy_score(svc.predict(X_select),y_res)}")
print(f"SVC test accuracy: {accuracy_score(svc.predict(X_select_test),y_test)}")
#Randomized Search CV
# param_dist={
#     'C':[0.1,0.3,0.5,1,2],
# }
# tree = LinearSVC() 
# tree_cv = RandomizedSearchCV(tree, param_dist, cv = 5) 
# tree_cv.fit(X_select, y_res) 
# print("Tuned Linear SVC Parameters: {}".format(tree_cv.best_params_)) 
# print("Best score is {}".format(tree_cv.best_score_))
 
#RANDOM FOREST
parameters={
    'n_estimators' : [50,100,200],
    'max_depth' : [10,40,None],
    'min_samples_leaf': [1, 4],
    'min_samples_split': [2,20,50,150,200,400,600]   
}
rf=RandomForestClassifier()
rf_cv = RandomizedSearchCV(rf, parameters,cv=5)
rf_cv.fit(X_select, y_res)
print("Tuned Random Forest Parameters: {}".format(rf_cv.best_params_)) 
print("Best score is {}".format(rf_cv.best_score_))
