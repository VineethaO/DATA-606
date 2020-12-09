#!/usr/bin/env python
# coding: utf-8

# ## Deliverable3:

# In[1]:


# Check the path and get the list of source files to loop through.

import os
os.getcwd()


# In[2]:


os.chdir('./Dataset/Data')
os.getcwd()


# In[3]:


A = os.listdir(os.getcwd())


# In[4]:


# Import all the required libraries

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
import re
from datetime import date
import twython
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Function to get the word count of the source files

Files = []
WordCount = []
DATE = []
MEDIUM = []
for filename in A:
    with open(filename,'rt', encoding = 'utf8') as file:
        text = file.read()
        words = text.split()
        NumOfWords = len(words)
        Files.append(filename)
        WordCount.append(NumOfWords)
        
        DateMatch = re.search("([0-9]{6})", filename)
        Date = DateMatch.group()
        FullDate = date(int(Date[2:]), int(Date[:2]), 1)
        DATE.append(FullDate)
        
        MediumName = re.search("[a-zA-Z]+", filename)
        Medium = MediumName.group()
        MEDIUM.append(Medium)


# In[6]:


# Create a dataframe with the word count data

data = zip(Files, WordCount, DATE, MEDIUM)
df_data = pd.DataFrame(list(data), columns= ['FileName', 'WordCount', 'Date', 'Medium'])


# In[7]:


# Loop through the text files, preprocess and find the sentiment of each of the text files

dates = []
media = []
possentiments = []
negsentiments = []
comsentiments = []

for filename in A:
    with open(filename,'rt', encoding = 'utf8') as file:
        text = file.read()
        
        DateMatch = re.search("([0-9]{6})", filename)
        Date = DateMatch.group()
        FullDate = date(int(Date[2:]), int(Date[:2]), 1)
        dates.append(FullDate)
        
        MediumName = re.search("[a-zA-Z]+", filename)
        Medium = MediumName.group()
        media.append(Medium)
        
        # split into words
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
        
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        
        # remove punctuation from each word
        import string
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        
        # filter out stop words
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        
        #lemmatization
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        CleanedText = ' '.join(words)
        
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        possentiment = sid.polarity_scores(CleanedText)['pos']
        negsentiment = sid.polarity_scores(CleanedText)['neg']
        comsentiment = sid.polarity_scores(CleanedText)['compound']
        possentiments.append(possentiment)
        negsentiments.append(negsentiment)
        comsentiments.append(comsentiment)


# In[8]:


# Zip the fields together to create pandas dataframe

details = zip(dates, media, possentiments, negsentiments, comsentiments)


# In[9]:


# Create a dataframe for all the text files including the sentiment.

import pandas as pd

df = pd.DataFrame(list(details), columns= ['Date', 'Medium', 'PositiveSentiment', 'NegativeSentiment', 'CompoundSentiment'])
df


# In[10]:


# Function to preprocess and find the sentiment.

def CalculateSentiment(text):
    
    # split into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
        
    # convert to lower case
    tokens = [w.lower() for w in tokens]
        
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
        
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
        
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    
    #lemmatization
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    CleanedText = ' '.join(words)
        
    sentiment = sid.polarity_scores(CleanedText)
    return sentiment


# In[11]:


# Calculate the sentiment region wise.

Chicagotext = ''
Denvertext = ''
Facebooktext = ''
LAtext = ''
NYtext = ''
Twittertext = ''
Washingtontext = ''
Result=[]
for filename in A:
    if re.search("[a-zA-Z]+", filename).group() == 'ChicagoTribune':
        with open(filename,'rt', encoding = 'utf8') as file:
            text = file.read()
            Chicagotext = Chicagotext + '\n' + text
    elif re.search("[a-zA-Z]+", filename).group() == 'DENVERPOST':
        with open(filename,'rt', encoding = 'utf8') as file:
            text = file.read()
            Denvertext = Denvertext + '\n' + text
    elif re.search("[a-zA-Z]+", filename).group() == 'Facebook':
        with open(filename,'rt', encoding = 'utf8') as file:
            text = file.read()
            Facebooktext = Facebooktext + '\n' + text
    elif re.search("[a-zA-Z]+", filename).group() == 'LATimes':
        with open(filename,'rt', encoding = 'utf8') as file:
            text = file.read()
            LAtext = LAtext + '\n' + text
    elif re.search("[a-zA-Z]+", filename).group() == 'NYTimes':
        with open(filename,'rt', encoding = 'utf8') as file:
            text = file.read()
            NYtext = NYtext + '\n' + text
    elif re.search("[a-zA-Z]+", filename).group() == 'Twitter':
        with open(filename,'rt', encoding = 'utf8') as file:
            text = file.read()
            Twittertext = Twittertext + '\n' + text
    else:
        with open(filename,'rt', encoding = 'utf8') as file:
            text = file.read()
            Washingtontext = Washingtontext + '\n' + text
    
my_dict = {'Chicago': CalculateSentiment(Chicagotext), 'Denver': CalculateSentiment(Denvertext),
           'LA': CalculateSentiment(LAtext), 'NY': CalculateSentiment(NYtext), 'Washington DC': CalculateSentiment(Washingtontext)}


# In[12]:


# Check the wordcount for the cleaned text of each region/social media platform.

print(len(Chicagotext.split())) #22517
print(len(Denvertext.split())) #28086
print(len(Facebooktext.split())) #15024
print(len(LAtext.split())) #31687
print(len(NYtext.split())) #26506
print(len(Twittertext.split())) #113195
print(len(Washingtontext.split())) #34921


# In[13]:


my_dict


# In[14]:


# Create the regionwise dataframe

df_Regionwise = pd.DataFrame(my_dict).transpose()
df_Regionwise.index.names = ['Region']
df_Regionwise.reset_index(level = 0, inplace = True)
df_Regionwise


# In[15]:


# Create the medium wise dataframe

DPtext = Chicagotext + '\n' + Denvertext +'\n' +  LAtext + '\n' + NYtext + '\n' + Washingtontext
SMtext = Facebooktext + '\n' + Twittertext
my_dict1 = {'DigitalPrintMedia': CalculateSentiment(DPtext), 'SocialMedia': CalculateSentiment(SMtext)}
df_Mediumwise = pd.DataFrame(my_dict1).transpose()
df_Mediumwise.index.names = ['Medium']
df_Mediumwise.reset_index(level = 0, inplace = True)
df_Mediumwise


# In[16]:


# Check the wordcount for the cleaned text of each medium.

print(len(DPtext.split())) #143717
print(len(SMtext.split())) #128219


# In[17]:


# Import visualization libraries

import chart_studio.plotly as py
import plotly.graph_objs as go
import pandas as pd


# In[18]:


# Create dataframe for monthwise data of Chicago region 

df_CT = df.loc[df['Medium'] == 'ChicagoTribune']
df_CT


# In[19]:


# Create dataframe for monthwise data of NewYork region 

df_NY = df.loc[df['Medium'] == 'NYTimes']
df_NY


# In[20]:


# Create dataframe for monthwise data of LA region 

df_LA = df.loc[df['Medium'] == 'LATimes']
df_LA


# In[21]:


# Create dataframe for monthwise data of Denver region 

df_Dv = df.loc[df['Medium'] == 'DENVERPOST']
df_Dv


# In[22]:


# Create dataframe for monthwise data of Washington region 

df_Wa = df.loc[df['Medium'] == 'WashingtonPost']
df_Wa


# In[23]:


# Create dataframe for monthwise data of Twitter

df_Tw = df.loc[df['Medium'] == 'Twitter']
df_Tw


# In[24]:


# Create dataframe for monthwise data of Facebook

df_Fb = df.loc[df['Medium'] == 'Facebook']
df_Fb


# In[25]:


# Prepare the data to train the machine learning models.
# Chicago dataframe

dfCT1 = df_CT[['Date','CompoundSentiment']]


# In[26]:


dfCT1


# In[27]:


# Fix the target values based on compound sentiment.

dfCT1 = dfCT1.assign(label = 0)
dfCT1.loc[dfCT1['CompoundSentiment'] > 0.2, 'label'] = 1
dfCT1.loc[dfCT1['CompoundSentiment'] < -0.2, 'label'] = -1
dfCT1


# In[28]:


dfCT1.set_index('Date', inplace=True)
dfCT1


# In[29]:


# Add days from start to use as X variable

dfCT1['days_from_start'] = (dfCT1.index - dfCT1.index[0]).days; dfCT1


# In[30]:


# Define X and y values.

X = dfCT1['days_from_start'].values.reshape(-1, 1)
y = dfCT1['label'].values


# In[31]:


# Prepare NewYork region data to train the ML models.

dfNY1 = df_NY[['Date','CompoundSentiment']]


# In[32]:


dfNY1 = dfNY1.assign(label = 0)
dfNY1.loc[dfNY1['CompoundSentiment'] > 0.2, 'label'] = 1
dfNY1.loc[dfNY1['CompoundSentiment'] < -0.2, 'label'] = -1
dfNY1


# In[33]:


dfNY1.set_index('Date', inplace=True)
dfNY1


# In[34]:


dfNY1['days_from_start'] = (dfNY1.index - dfNY1.index[0]).days; dfNY1


# In[35]:


# Define X and y values for NewYork data

X1 = dfNY1['days_from_start'].values.reshape(-1, 1)
y1 = dfNY1['label'].values


# In[36]:


# Prepare Denver region data to train the ML models.

dfDv1 = df_Dv[['Date','CompoundSentiment']]
dfDv1 = dfDv1.assign(label = 0)
dfDv1.loc[dfDv1['CompoundSentiment'] > 0.2, 'label'] = 1
dfDv1.loc[dfDv1['CompoundSentiment'] < -0.2, 'label'] = -1
dfDv1


# In[37]:


dfDv1.set_index('Date', inplace=True)
dfDv1['days_from_start'] = (dfDv1.index - dfDv1.index[0]).days; dfDv1


# In[38]:


# Define X and y values for Denver data

X2 = dfDv1['days_from_start'].values.reshape(-1, 1)
y2 = dfDv1['label'].values


# In[39]:


# Prepare LA region data to train the ML models.

dfLA1 = df_LA[['Date','CompoundSentiment']]
dfLA1 = dfLA1.assign(label = 0)
dfLA1.loc[dfLA1['CompoundSentiment'] > 0.2, 'label'] = 1
dfLA1.loc[dfLA1['CompoundSentiment'] < -0.2, 'label'] = -1
dfLA1.set_index('Date', inplace=True)
dfLA1['days_from_start'] = (dfLA1.index - dfLA1.index[0]).days; dfLA1


# In[40]:


# Define X and y values for LA data

X3 = dfLA1['days_from_start'].values.reshape(-1, 1)
y3 = dfLA1['label'].values


# In[41]:


# Prepare Washington region data to train the ML models.

dfWa1 = df_Wa[['Date','CompoundSentiment']]
dfWa1 = dfWa1.assign(label = 0)
dfWa1.loc[dfWa1['CompoundSentiment'] > 0.2, 'label'] = 1
dfWa1.loc[dfWa1['CompoundSentiment'] < -0.2, 'label'] = -1
dfWa1.set_index('Date', inplace=True)
dfWa1['days_from_start'] = (dfWa1.index - dfWa1.index[0]).days; dfWa1


# In[42]:


# Define X and y values for Washington data

X4 = dfWa1['days_from_start'].values.reshape(-1, 1)
y4 = dfWa1['label'].values


# In[43]:


# Prepare Twitter data to train the ML models.

dfTw1 = df_Tw[['Date','CompoundSentiment']]
dfTw1 = dfTw1.assign(label = 0)
dfTw1.loc[dfTw1['CompoundSentiment'] > 0.9998, 'label'] = 1
dfTw1.loc[dfTw1['CompoundSentiment'] <= 0.9998, 'label'] = -1
dfTw1.set_index('Date', inplace=True)
dfTw1['days_from_start'] = (dfTw1.index - dfTw1.index[0]).days; dfTw1


# In[44]:


# Define X and y values for Twitter data

X5 = dfTw1['days_from_start'].values.reshape(-1, 1)
y5 = dfTw1['label'].values


# In[45]:


# Prepare Facebook data to train the ML models.

dfFb1 = df_Fb[['Date','CompoundSentiment']]
dfFb1 = dfFb1.assign(label = 0)
dfFb1.loc[dfFb1['CompoundSentiment'] > 0.997, 'label'] = 1 
dfFb1.loc[dfFb1['CompoundSentiment'] <= 0.997, 'label'] = -1
dfFb1.set_index('Date', inplace=True)
dfFb1['days_from_start'] = (dfFb1.index - dfFb1.index[0]).days; dfFb1


# In[46]:


# Define X and y values for Facebook data

X6 = dfFb1['days_from_start'].values.reshape(-1, 1)
y6 = dfFb1['label'].values


# ## Training Machine Learning models

# ### Logistic Regression:

# In[47]:


# Split the data into train 60% and test 40%.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)


# In[48]:


# Import Logistic Regression and train.

from sklearn import linear_model
modellr = linear_model.LogisticRegression(C=0.05).fit(X_train, y_train) # smaller C specify stronger regularization.


# In[49]:


# Predict using Logistic Regression

y_pred1 = modellr.predict(X_test)


# In[50]:


# Create the confusion matrix and find accuracy

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, y_pred1)
c_matrix = confusion_matrix(y_test, y_pred1)
print("The model accuracy is", accuracy )


# In[51]:


# Use Seaborn heatmap to show TP, FP, TN, FN values

import seaborn as sns
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_percentages = ["{0:.2%}".format(value) for value in c_matrix.flatten()/np.sum(c_matrix)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_percentages)]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(c_matrix, annot=labels, fmt="", cmap='Blues')
plt.show()


# ### Support Vector Machines:

# In[52]:


# Import SVM Classifier and train.

from sklearn.svm import SVC
classifierSV = SVC(kernel='linear')
classifierSV.fit(X_train, y_train)


# In[53]:


# Predict using SVM

y_pred2 = classifierSV.predict(X_test)


# In[54]:


# Calculate the model accuracy

print("The model accuracy is", accuracy_score(y_test, y_pred2) )


# In[55]:


# Create Classification report for SVM model

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred2, target_names=['Class 1', 'Class -1']))


# ### KNeighborsClassifier

# In[56]:


# Split the data into train and test.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size = 0.3, random_state = 18)


# In[57]:


# Import the KNeighborsClassifier class from sklearn

from sklearn.neighbors import KNeighborsClassifier

#import metrics model to check the accuracy
from sklearn import metrics

#Try running from k=1 through 9 and record testing accuracy
k_range = range(1,9)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    y_pred3 = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred3)
    scores_list.append((metrics.accuracy_score(y_test,y_pred3)))


# In[58]:


#plot the relationship between K and the testing accuracy

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.plot(k_range,scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.title('Accuracy plot of KNN')
plt.show()


# In[59]:


# Train the KNN Classifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)


# In[60]:


# Predict using the KNN Classifier

y_pred3 = knn.predict(X_test)


# In[61]:


# Calculate the model accuracy

print("The model accuracy is", accuracy_score(y_test, y_pred3))


# ### Random Forests:

# In[62]:


# Split the data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size = 0.40, random_state = 15)


# In[63]:


#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clfRF=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfRF.fit(X_train,y_train)

y_pred4=clfRF.predict(X_test)


# In[64]:


# Calculate the model accuracy

print("The model accuracy is", accuracy_score(y_test, y_pred4))


# In[65]:


# Provide the ROC curve. What is the area under the curve?

from sklearn.metrics import roc_curve, roc_auc_score
 
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred4) # Creating true and false positive rates
plt.title('Receiver Operating Characteristic - Random Forest')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('roc_auc_score for Random Forest model is ', roc_auc_score(y_test, y_pred4))


# ### Long Short Term Memory (LSTM):

# In[66]:


# Split the data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size = 0.50, random_state = 2)


# In[67]:


# Import the required libraries

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


# In[68]:


# Scale the data using MinMaxscaler

Xscaler = MinMaxScaler(feature_range=(0, 1)) # scale so that all the X data will range from 0 to 1
Xscaler.fit(X_train)
scaled_X_train = Xscaler.transform(X_train)
print(X_train.shape)


# In[69]:


y_train.shape


# In[70]:


y_train = y_train.reshape(-1,1)


# In[71]:


Yscaler = MinMaxScaler(feature_range=(0, 1))
Yscaler.fit(y_train)
scaled_y_train = Yscaler.transform(y_train)
print(scaled_y_train.shape)


# In[72]:


scaled_y_train = scaled_y_train.reshape(-1) # remove the second dimention from y so the shape changes from (n,1) to (n,)
print(scaled_y_train.shape)


# In[73]:


X_train.shape[1]


# In[74]:


# Define the inputs to LSTM model and initialize the TimeseriesGenerator

n_input = 3 #how many samples/rows/timesteps to look in the past in order to forecast the next sample
n_features= X_train.shape[1] # how many predictors/Xs/features we have to predict y
b_size = 6 # Number of timeseries samples in each batch
generator = TimeseriesGenerator(scaled_X_train, scaled_y_train, length=n_input, batch_size=b_size)
n_outputs = 1
print(generator[0][0].shape)


# In[75]:


# Build the LSTM model

model = Sequential()
model.add(LSTM(100, input_shape=(n_input,n_features)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
model.summary()


# In[76]:


# Fit the model

model.fit(generator,epochs=50)


# In[77]:


# Loss plot

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()


# In[78]:


# Scale the test data and initialize the TimeseriesGenerator

scaled_X_test = Xscaler.transform(X_test)
test_generator = TimeseriesGenerator(scaled_X_test, np.zeros(len(X_test)), length=n_input, batch_size=b_size)
print(test_generator[0][0].shape)


# In[79]:


scaled_X_test


# In[80]:


# Predict the data

y_pred_scaled = model.predict(test_generator)
y_pred5 = Yscaler.inverse_transform(y_pred_scaled)


# In[81]:


y_pred_scaled


# In[82]:


y_pred5


# In[83]:


def get_pred(x):
    if x >= 0.25:
        return 1
    return -1


# In[84]:


# Convert the predicted data from scaled version to normal

y_prediction = [get_pred(row) for row in y_pred5]


# In[85]:


y_prediction


# In[86]:


y_test[n_input:]


# In[87]:


# Calculate the model accuracy

print("The model accuracy is", round(accuracy_score(y_test[n_input:], y_prediction),2) )


# ### Forecasting sentiment for Washington region:

# In[88]:


# Split the Washington data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size = 0.40, random_state = 0)


# In[89]:


#Training Random Forest Model with Washington Region data.
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clfRF=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfRF.fit(X_train,y_train)


# In[90]:


# Create a dataframe with future timesteps

dfWa1_ft = pd.DataFrame({'date':pd.to_datetime(['2020-11-01', '2020-12-01','2021-01-01'])})
dfWa1_ft


# In[91]:


dfWa1_ft['date'] = pd.to_datetime(dfWa1_ft['date']).dt.date


# In[92]:


dfWa1_ft.set_index('date', inplace=True)
dfWa1_ft


# In[93]:


dfWa1_ft['days_from_start'] = (dfWa1_ft.index - dfWa1.index[0]).days
dfWa1_ft


# In[94]:


dfWa1_ft['prediction'] =clfRF.predict(dfWa1_ft['days_from_start'].values.reshape(-1, 1))


# In[95]:


dfWa1_ft


# In[96]:


dfWa1 = dfWa1.sort_index()
dfWa1


# In[97]:


# Concatenate with the previous dataframe

dfWa_final = pd.concat([dfWa1, dfWa1_ft], axis=1, sort=True)
dfWa_final


# In[98]:


# Plot the Sentiment prediction for Washington Region

fig_Wa = go.Figure()
fig_Wa.add_trace(go.Bar(x=dfWa_final.index, y=dfWa_final.label, name='Actual'))
fig_Wa.add_trace(go.Bar(x=dfWa_final.index, y=dfWa_final.prediction, name='Prediction'))

fig_Wa.update_layout(title='Sentiment Prediction for Washington Region')
fig_Wa.show()


# ### Forecasting sentiment for Chicago region:

# In[99]:


# Split the Chicago data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)


# In[100]:


#Training Random Forest Model with Chicago region data.
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clfRF1=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfRF1.fit(X_train,y_train)


# In[101]:


# Create a dataframe with future timesteps and concatenate it with the previous dataframe

dfCT1_ft = pd.DataFrame({'date':pd.to_datetime(['2020-11-01', '2020-12-01','2021-01-01'])})
dfCT1_ft['date'] = pd.to_datetime(dfCT1_ft['date']).dt.date
dfCT1_ft.set_index('date', inplace=True)
dfCT1_ft['days_from_start'] = (dfCT1_ft.index - dfCT1.index[0]).days
dfCT1_ft['prediction'] =clfRF1.predict(dfCT1_ft['days_from_start'].values.reshape(-1, 1))
dfCT1 = dfCT1.sort_index()
dfCT_final = pd.concat([dfCT1, dfCT1_ft], axis=1, sort=True)
dfCT_final


# In[102]:


# Plot the Sentiment prediction for Chicago Region

fig_CT = go.Figure()
fig_CT.add_trace(go.Bar(x=dfCT_final.index, y=dfCT_final.label, name='Actual'))
fig_CT.add_trace(go.Bar(x=dfCT_final.index, y=dfCT_final.prediction, name='Prediction'))

fig_CT.update_layout(title='Sentiment Prediction for Chicago Region')
fig_CT.show()


# ### Forecasting sentiment for NewYork region:

# In[103]:


# Split the NewYork data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.40, random_state = 0)


# In[104]:


#Training Random Forest Model with NewYork region data.
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clfRF2=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfRF2.fit(X_train,y_train)


# In[105]:


# Create a dataframe with future timesteps and concatenate it with the previous dataframe

dfNY1_ft = pd.DataFrame({'date':pd.to_datetime(['2020-11-01', '2020-12-01','2021-01-01'])})
dfNY1_ft['date'] = pd.to_datetime(dfNY1_ft['date']).dt.date
dfNY1_ft.set_index('date', inplace=True)
dfNY1_ft['days_from_start'] = (dfNY1_ft.index - dfNY1.index[0]).days
dfNY1_ft['prediction'] =clfRF2.predict(dfNY1_ft['days_from_start'].values.reshape(-1, 1))
dfNY1 = dfNY1.sort_index()
dfNY_final = pd.concat([dfNY1, dfNY1_ft], axis=1, sort=True)
dfNY_final


# In[106]:


# Plot the Sentiment prediction for NewYork Region

fig_NY = go.Figure()
fig_NY.add_trace(go.Bar(x=dfNY_final.index, y=dfNY_final.label, name='Actual'))
fig_NY.add_trace(go.Bar(x=dfNY_final.index, y=dfNY_final.prediction, name='Prediction'))

fig_NY.update_layout(title='Sentiment Prediction for NewYork Region')
fig_NY.show()


# ### Forecasting sentiment for Denver region:

# In[107]:


# Split the Denver data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.40, random_state = 0)


# In[108]:


#Training Random Forest Model with Denver region data.
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clfRF3=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfRF3.fit(X_train,y_train)


# In[109]:


# Create a dataframe with future timesteps and concatenate it with the previous dataframe

dfDv1_ft = pd.DataFrame({'date':pd.to_datetime(['2020-11-01', '2020-12-01','2021-01-01'])})
dfDv1_ft['date'] = pd.to_datetime(dfDv1_ft['date']).dt.date
dfDv1_ft.set_index('date', inplace=True)
dfDv1_ft['days_from_start'] = (dfDv1_ft.index - dfDv1.index[0]).days
dfDv1_ft['prediction'] =clfRF3.predict(dfDv1_ft['days_from_start'].values.reshape(-1, 1))
dfDv1 = dfDv1.sort_index()
dfDv_final = pd.concat([dfDv1, dfDv1_ft], axis=1, sort=True)
dfDv_final


# In[110]:


# Plot the Sentiment prediction for Denver Region

fig_Dv = go.Figure()
fig_Dv.add_trace(go.Bar(x=dfDv_final.index, y=dfDv_final.label, name='Actual'))
fig_Dv.add_trace(go.Bar(x=dfDv_final.index, y=dfDv_final.prediction, name='Prediction'))

fig_Dv.update_layout(title='Sentiment Prediction for Denver Region')
fig_Dv.show()


# ### Forecasting sentiment for LosAngeles region:

# In[111]:


# Split the LA data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size = 0.40, random_state = 0)


# In[112]:


#Training Random Forest Model with LosAngeles region data.
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clfRF4=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfRF4.fit(X_train,y_train)


# In[113]:


# Create a dataframe with future timesteps and concatenate it with the previous dataframe

dfLA1_ft = pd.DataFrame({'date':pd.to_datetime(['2020-11-01', '2020-12-01','2021-01-01'])})
dfLA1_ft['date'] = pd.to_datetime(dfLA1_ft['date']).dt.date
dfLA1_ft.set_index('date', inplace=True)
dfLA1_ft['days_from_start'] = (dfLA1_ft.index - dfLA1.index[0]).days
dfLA1_ft['prediction'] =clfRF4.predict(dfLA1_ft['days_from_start'].values.reshape(-1, 1))
dfLA1 = dfLA1.sort_index()
dfLA_final = pd.concat([dfLA1, dfLA1_ft], axis=1, sort=True)
dfLA_final


# In[114]:


# Plot the Sentiment prediction for LA Region

fig_LA = go.Figure()
fig_LA.add_trace(go.Bar(x=dfLA_final.index, y=dfLA_final.label, name='Actual'))
fig_LA.add_trace(go.Bar(x=dfLA_final.index, y=dfLA_final.prediction, name='Prediction'))

fig_LA.update_layout(title='Sentiment Prediction for Los Angeles Region')
fig_LA.show()


# ### Forecasting sentiment for Twitter data:

# In[115]:


# Split the Twitter data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X5, y5, test_size = 0.40, random_state = 0)


# In[116]:


#Training Random Forest Model with Twitter data.
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clfRF5=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfRF5.fit(X_train,y_train)


# In[117]:


# Create a dataframe with future timesteps and concatenate it with the previous dataframe

dfTw1_ft = pd.DataFrame({'date':pd.to_datetime(['2020-11-01', '2020-12-01','2021-01-01'])})
dfTw1_ft['date'] = pd.to_datetime(dfTw1_ft['date']).dt.date
dfTw1_ft.set_index('date', inplace=True)
dfTw1_ft['days_from_start'] = (dfTw1_ft.index - dfTw1.index[0]).days
dfTw1_ft['prediction'] =clfRF5.predict(dfTw1_ft['days_from_start'].values.reshape(-1, 1))
dfTw1 = dfTw1.sort_index()
dfTw_final = pd.concat([dfTw1, dfTw1_ft], axis=1, sort=True)
dfTw_final


# In[118]:


# Plot the Sentiment prediction for Twitter data

fig_Tw = go.Figure()
fig_Tw.add_trace(go.Bar(x=dfTw_final.index, y=dfTw_final.label, name='Actual'))
fig_Tw.add_trace(go.Bar(x=dfTw_final.index, y=dfTw_final.prediction, name='Prediction'))

fig_Tw.update_layout(title='Sentiment Prediction for Twitter data')
fig_Tw.show()


# ### Forecasting sentiment for Facebook data:

# In[119]:


# Split the Facebook data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X6, y6, test_size = 0.40, random_state = 0)


# In[120]:


#Training Random Forest Model with Facebook data.
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clfRF6=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfRF6.fit(X_train,y_train)


# In[121]:


# Create a dataframe with future timesteps and concatenate it with the previous dataframe

dfFb1_ft = pd.DataFrame({'date':pd.to_datetime(['2020-11-01', '2020-12-01','2021-01-01'])})
dfFb1_ft['date'] = pd.to_datetime(dfFb1_ft['date']).dt.date
dfFb1_ft.set_index('date', inplace=True)
dfFb1_ft['days_from_start'] = (dfFb1_ft.index - dfFb1.index[0]).days
dfFb1_ft['prediction'] =clfRF6.predict(dfFb1_ft['days_from_start'].values.reshape(-1, 1))
dfFb1 = dfFb1.sort_index()
dfFb_final = pd.concat([dfFb1, dfFb1_ft], axis=1, sort=True)
dfFb_final


# In[122]:


# Plot the Sentiment prediction for Facebook data

fig_Fb = go.Figure()
fig_Fb.add_trace(go.Bar(x=dfFb_final.index, y=dfFb_final.label, name='Actual'))
fig_Fb.add_trace(go.Bar(x=dfFb_final.index, y=dfFb_final.prediction, name='Prediction'))

fig_Fb.update_layout(title='Sentiment Prediction for Facebook data')
fig_Fb.show()


# In[ ]:




