# Fraud-Recruitment-Detector

This project can be used to detect the fake job vacancy which is being posted by scammers for their own monetary benifit. The proposed model uses diffrent machine Learning algorithms to detect Fraud job vacancy.
For real time job prediction ,data required can be collected through diffrent job posting website,Linkedin,carrer page of Tech companies. Web scrapper is designed for collection of data from diffrent sources and then after preprocessing , stored in our database through Live streaming Technique.
For modelling purpose EMSCAD dataset is used which is freely available on Kaggle.
The dataset description is as follows:
![image](https://user-images.githubusercontent.com/85219308/125249368-ca6d2880-e312-11eb-9f44-089249b97ef0.png)

After importing dataset the next step was looking inside the data which was being generated using Pandas Profiling report.The key points observed was:
1. Job id is uniform and only an identification of job vacancy.
2. Some salary range is written in range, some in the value, and some is missing. However, there are very many different range.
3. Company profile and Description is text data.
4. Categorical data like employment type, required education, and required experience have many missing values.
5. Dataset is unbalanced, the percentage of real job vacancy and fake job vacancy is 20:1.
To understand more about the data Exploratory Data Analysis was done. This was the result gathered:
# Exploratory Data Analysis
![image](https://user-images.githubusercontent.com/85219308/125249644-0f915a80-e313-11eb-838d-fddc7ca32f44.png)
The graph shows that if a company provides telecommuting it is more probable to be a fraud as compared to the company which does not provide it.
![image](https://user-images.githubusercontent.com/85219308/125250015-6c8d1080-e313-11eb-8d30-cf902be2296c.png)
The next feature is do company ask any questions before hiring or in other words take interview.A company which does not take interview is more probable to be a fraud as compared to company which takes interview.

![image](https://user-images.githubusercontent.com/85219308/125250261-a78f4400-e313-11eb-9e58-d111ba0a63f4.png)
The next important feature is employment type the company which gives us part time job is more prone to be a fraud compared to any other employment type.
![image](https://user-images.githubusercontent.com/85219308/125250353-c2fa4f00-e313-11eb-8949-cf7c47206e18.png)
The next important feature is require education.if a company hires on the basis of only high school education .It is more probable to be a fraud as compared to hiring on any other educational basis.
After performing EDA the next step conducted is Feature Engineering.
# Feature Engineering
The data needed to be processed first before inserted into machine learning model. First all of the feature will be made into one text. The text will then undergo various text cleaning like normalization. Finally the feature from data will be extracted with TfIdfVectorizer.
1. Changing feature before appending into one text
First, all of the boolean value needed to be converted to text so that TfIdfVectorizer could differentiate which boolean belongs to which. After that, categorical data needed to be converted so that the information that the categorical data is indeed categorical and not another text still holds. This could be done by appending column name and changing “_” from space.
2. Appending into one text
The appending part is quite easy, just append with code below. However, job id will be dropped because it contains no information and fraudulent column will also be dropped because fraudulent is the target column that’s going to be predicted.
3. Text Cleaning
There are some text cleaning method that needs to be done before we extract the feature from the text.
a. Lowering the case
Lowering the case is important since capital case in english is only used for first word in sentence and/or name. Since we expect same predictive power from lower or capital case, we will lower all the word in text.
b. Lemmatization and Removing Stopwords
Lemmatization is the process of converting a word into its base form. For example, feett will be converted to foott. Lemmatization could be done by the following code using spacy. Tag argument is set to be true to differentiate which one is noun, verb, etc. Parse argument is set to be false to make the model faster. Finally, entity argument is set to be true to differentiate named entity(like Apple, Google, etc) from regular noun. After that, stopwords(“The”,”in”) will be removed since it doesn’t have predictive power. However, “no” and “not” will be kept.
c. Removing Special Character and Punctuation
Special character like $,%,#, and punctuation like ; and ? will be removed because no predictive power is expected and will just introduce noice to the text. Because of the previous result, deleting punctuation will create double space in the data, therefore double space will be converted into single space. Possessive “s” like in he’s will also created “ ‘s ”. ‘s will also be deleted.
d. Combining
It’s time to apply the text cleaning into the text. Since there are 17880 data, it will take times to finish.
4. Feature Extraction using TfIdfVectorizer
Next, text that has been cleaned will be transformed into matrix that machine learning model could process. One way of doing that is applying TfIdfVectorizer. This method will count the importance of words in a document and the rarity of the word itself. ngram_range argument is used to specify what n-grams value to be extracted. Value of (1,2) means extract both unigrams (ex : box) and bigrams (ex : black box).
5. Splitting into Train and Test data
Train data will be used to train the model and test data will be used to check the accuracy of the prediction. The data will be split 80–20.
6. Oversampling
As observed in the beginning, the target class is very imbalance. Hence, oversampling method, ADASYN, will be used to prevent class imbalance.
# Modelling
1. Feature Selection
Various model will be trained. In order to minimize training time, feature selection will be conducted first. Feature selection will be conducted by selecting only important feature by training the data first using Linear SVC.
2. Base model
Base Random Forest, LinearSVC, Gradient Boosting, and XGBoost model will be trained without tuning the hyper-parameter. The performance which each model will be measured by accuracy score since for this particular model it is assumed that the weights of false positive and false negative results is same.
Linear SVC gives the best accuracy score with almost 99% accuracy. Let’s take a look into the classification metrics.
The precision score of the model is lower than the rest because of the amount of false negative. Recall and Accuracy score gives satisfying result with score greater than 95%. Finally, let’s look at how the Linear SVC model classify the results by looking at top 10 most important feature
![image](https://user-images.githubusercontent.com/85219308/125252113-a19a6280-e315-11eb-8ded-2836e422bbde.png)
