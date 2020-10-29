#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[2]:


os.chdir('./Dataset/Data')
os.getcwd()


# In[3]:


A = os.listdir(os.getcwd())


# In[4]:


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


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import cufflinks as cf
cf.go_offline()


# In[6]:


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


# In[7]:


data = zip(Files, WordCount, DATE, MEDIUM)
df_data = pd.DataFrame(list(data), columns= ['FileName', 'WordCount', 'Date', 'Medium'])


# In[8]:


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


# In[9]:


details = zip(dates, media, possentiments, negsentiments, comsentiments)


# In[10]:


import pandas as pd

df = pd.DataFrame(list(details), columns= ['Date', 'Medium', 'PositiveSentiment', 'NegativeSentiment', 'CompoundSentiment'])
df


# In[11]:


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


# In[12]:


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


# In[13]:


print(len(Chicagotext.split())) #22517
print(len(Denvertext.split())) #28086
print(len(Facebooktext.split())) #15024
print(len(LAtext.split())) #31687
print(len(NYtext.split())) #26506
print(len(Twittertext.split())) #113195
print(len(Washingtontext.split())) #34921


# In[14]:


my_dict


# In[15]:


df_Regionwise = pd.DataFrame(my_dict).transpose()
df_Regionwise.index.names = ['Region']
df_Regionwise.reset_index(level = 0, inplace = True)
df_Regionwise


# In[16]:


DPtext = Chicagotext + '\n' + Denvertext +'\n' +  LAtext + '\n' + NYtext + '\n' + Washingtontext
SMtext = Facebooktext + '\n' + Twittertext
my_dict1 = {'DigitalPrintMedia': CalculateSentiment(DPtext), 'SocialMedia': CalculateSentiment(SMtext)}
df_Mediumwise = pd.DataFrame(my_dict1).transpose()
df_Mediumwise.index.names = ['Medium']
df_Mediumwise.reset_index(level = 0, inplace = True)
df_Mediumwise


# In[17]:


print(len(DPtext.split())) #143717
print(len(SMtext.split())) #128219


# In[19]:


# Region-wise trend of people's sentiment towards environmental protection.

import plotly.express as px

fig1 = px.bar(df_Regionwise, x="Region", y= ['pos', 'neg'], barmode='group',
             title = "Region-wise trend of people's sentiment towards environmental protection")
fig1.show()


# In[32]:


# Generate html file

import plotly.io as pio

pio.write_html(fig1, file='Region-wise trend.html', auto_open=True)


# In[21]:


# Comparison of people's sentiment towards environmental protection between social media and Digital Print media

fig2 = px.bar(df_Mediumwise, x="Medium", y= ['neg', 'neu', 'pos'], barmode='group',
             title = "Digital Print media vs Social media - comparison of people's sentiment")
fig2.show()


# In[33]:


# Generate html file

pio.write_html(fig2, file='Digital Print media vs Social media.html', auto_open=True)


# In[22]:


import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
import pandas as pd


# In[23]:


us_state_abbrev = {
    'California': 'CA',
    'Connecticut': 'CT',
    'Colorado': 'CO',
    'Illinois': 'IL',
    'Maryland': 'MD',
    'New Jersey': 'NJ',
    'New York': 'NY',
    'Virginia': 'VA'
}


# In[24]:


RegionToStateMapping = {
    'LA': 'California',
    'NY': 'Connecticut',
    'Denver': 'Colorado',
    'Chicago': 'Illinois',
    'Washington':'Maryland',
    'NY': 'New Jersey',
    'NY': 'New York',
    'Washington DC': 'Virginia'
}


# In[25]:


df_Regionwise['State'] = df_Regionwise['Region'].map(RegionToStateMapping)


# In[26]:


df_Regionwise['Abbrev'] = df_Regionwise['State'].map(us_state_abbrev)


# In[27]:


df_Regionwise.head()


# In[28]:


map_data = dict(type='choropleth',
            locations=df_Regionwise['Abbrev'],
            locationmode='USA-states',
            colorscale='Reds',
            text=df_Regionwise['Region'],
            marker=dict(line=dict(color='rgb(255,0,0)', width=2)),
            z=df_Regionwise['neg'],
            colorbar=dict(title="Negative Sentiment")
           )


# In[29]:


map_layout = dict(title='Region wise variation of negative sentiment',
              geo=dict(scope='usa',
                         showlakes=True,
                         lakecolor='rgb(85,173,240)')
             )


# In[30]:


map_actual = go.Figure(data=[map_data], layout=map_layout)


# In[31]:


iplot(map_actual)


# In[34]:


# Generate html file

pio.write_html(map_actual, file='Trend of Negative Sentiment.html', auto_open=True)


# In[35]:


map_data1 = dict(type='choropleth',
            locations=df_Regionwise['Abbrev'],
            locationmode='USA-states',
            colorscale='Greens',
            text=df_Regionwise['Region'],
            marker=dict(line=dict(color='rgb(255,0,0)', width=2)),
            z=df_Regionwise['pos'],
            colorbar=dict(title="Positve Sentiment")
           )


# In[36]:


map_layout1 = dict(title='Region wise variation of positive sentiment',
              geo=dict(scope='usa',
                         showlakes=True,
                         lakecolor='rgb(85,173,240)')
             )


# In[39]:


map_actual1 = go.Figure(data=[map_data1], layout=map_layout1)


# In[40]:


iplot(map_actual1)


# In[41]:


# Generate html file

pio.write_html(map_actual1, file='Trend of Positive Sentiment.html', auto_open=True)


# In[42]:


df_CT = df.loc[df['Medium'] == 'ChicagoTribune']
df_CT


# In[43]:


df_CT['Date'] = pd.to_datetime(df_CT['Date'], format='%Y-%m-%d').sort_values().unique()


# In[44]:


df_CT.dtypes


# In[45]:


layout = dict(title = "Month wise trend of Chicago region",
              xaxis = dict(title = 'Regions'),
              yaxis = dict(title = 'Sentiment'),
              )
df_CT.iplot(kind = 'bar', x = 'Date', y = ['PositiveSentiment','NegativeSentiment'], layout = layout)


# In[46]:


df_NY = df.loc[df['Medium'] == 'NYTimes']
df_NY


# In[47]:


layout = dict(title = "Month wise trend of NewYork region",
              xaxis = dict(title = 'Regions'),
              yaxis = dict(title = 'Sentiment'),
              )
df_NY.iplot(kind = 'bar', x = 'Date', y = ['PositiveSentiment','NegativeSentiment'], layout = layout)


# In[48]:


df_LA = df.loc[df['Medium'] == 'LATimes']
df_LA


# In[49]:


layout = dict(title = "Month wise trend of LosAngeles region",
              xaxis = dict(title = 'Regions'),
              yaxis = dict(title = 'Sentiment'),
              )
df_LA.iplot(kind = 'bar', x = 'Date', y = ['PositiveSentiment','NegativeSentiment'], layout = layout)


# In[50]:


df_Dv = df.loc[df['Medium'] == 'DENVERPOST']
df_Dv


# In[51]:


layout = dict(title = "Month wise trend of Denver region",
              xaxis = dict(title = 'Regions'),
              yaxis = dict(title = 'Sentiment'),
              )
df_Dv.iplot(kind = 'bar', x = 'Date', y = ['PositiveSentiment','NegativeSentiment'], layout = layout)


# In[52]:


df_Wa = df.loc[df['Medium'] == 'WashingtonPost']
df_Wa


# In[53]:


layout = dict(title = "Month wise trend of people's sentiment in Washington DC region",
              xaxis = dict(title = 'Regions'),
              yaxis = dict(title = 'Sentiment'),
              )
df_Wa.iplot(kind = 'bar', x = 'Date', y = ['PositiveSentiment','NegativeSentiment'], layout = layout)


# In[55]:


import plotly.graph_objects as go

import pandas as pd

# create figure
fig3 = go.Figure()

# Add= trace
fig3.add_trace(go.Bar(name='Positive', x=df_CT['Date'], y=df_CT['PositiveSentiment']))
fig3.add_trace(go.Bar(name='Negative', x=df_CT['Date'], y=df_CT['NegativeSentiment']))
fig3.add_trace(go.Bar(name='Positive', x=df_Dv['Date'], y=df_Dv['PositiveSentiment']))
fig3.add_trace(go.Bar(name='Negative', x=df_Dv['Date'], y=df_Dv['NegativeSentiment']))
fig3.add_trace(go.Bar(name='Positive', x=df_LA['Date'], y=df_LA['PositiveSentiment']))
fig3.add_trace(go.Bar(name='Negative', x=df_LA['Date'], y=df_LA['NegativeSentiment']))
fig3.add_trace(go.Bar(name='Positive', x=df_NY['Date'], y=df_NY['PositiveSentiment']))
fig3.add_trace(go.Bar(name='Negative', x=df_NY['Date'], y=df_NY['NegativeSentiment']))
fig3.add_trace(go.Bar(name='Positive', x=df_Wa['Date'], y=df_Wa['PositiveSentiment']))
fig3.add_trace(go.Bar(name='Negative', x=df_Wa['Date'], y=df_Wa['NegativeSentiment']))



# Add dropdown
fig3.update_layout(
    updatemenus=[
        dict( 
            active = 0,
            buttons=list([
                dict(label = 'Chicago',
                  method = 'update',
                  args = [{'visible': [True, True, False, False, False, False, False, False, False, False]},
                          {'title': "Month wise trend of people's sentiment in Chicago region",
                           'showlegend':True}]),
               dict(label = 'Denver',
                  method = 'update',
                  args = [{'visible': [False, False, True, True, False, False, False, False, False, False]},
                          {'title': "Month wise trend of people's sentiment in Denver region",
                           'showlegend':True}]),
                dict(label = 'Los Angeles',
                  method = 'update',
                  args = [{'visible': [False, False, False, False, True, True, False, False, False, False]},
                          {'title': "Month wise trend of people's sentiment in Los Angeles region",
                           'showlegend':True}]),
                dict(label = 'NewYork',
                  method = 'update',
                  args = [{'visible': [False, False, False, False, False, False, True, True, False, False]},
                          {'title': "Month wise trend of people's sentiment in NewYork region",
                           'showlegend':True}]),
                dict(label = 'WashingtonDC',
                  method = 'update',
                  args = [{'visible': [False, False, False, False, False, False, False, False, True, True]},
                          {'title': "Month wise trend of people's sentiment in Washington DC region",
                           'showlegend':True}])
            ]),
            
        )
    ]
)

fig3.show()


# In[56]:


# Generate html file

pio.write_html(fig3, file='Monthwise Trend.html', auto_open=True)


# In[57]:


df_Tw = df.loc[df['Medium'] == 'Twitter']
df_Tw


# In[58]:


df_Fb = df.loc[df['Medium'] == 'Facebook']
df_Fb


# In[59]:


df_TWord = df_data[['Date','Medium','WordCount']].loc[df_data['Medium'] == 'Twitter'].reset_index()
df_TWord


# In[60]:


df_FbWord = df_data[['Date','Medium','WordCount']].loc[df_data['Medium'] == 'Facebook'].reset_index()
df_FbWord


# In[61]:


import plotly.graph_objects as go

import pandas as pd

# create figure
fig4 = go.Figure()

# Add= trace
fig4.add_trace(go.Bar(name='Positive', x=df_Tw['Date'], y=df_Tw['PositiveSentiment']))
fig4.add_trace(go.Bar(name='Negative', x=df_Tw['Date'], y=df_Tw['NegativeSentiment']))
fig4.add_trace(go.Bar(name='Positive', x=df_Fb['Date'], y=df_Fb['PositiveSentiment']))
fig4.add_trace(go.Bar(name='Negative', x=df_Fb['Date'], y=df_Fb['NegativeSentiment']))
fig4.add_trace(go.Bar(name='Twitter WordCount', x=df_TWord['Date'], y=df_TWord['WordCount']))
fig4.add_trace(go.Bar(name='Facebook WordCount', x=df_FbWord['Date'], y=df_FbWord['WordCount']))


# Add dropdown
fig4.update_layout(
    updatemenus=[
        dict( 
            active = 0,
            buttons=list([
                dict(label = 'Twitter',
                  method = 'update',
                  args = [{'visible': [True, True, False, False, False, False]},
                          {'title': "Month wise trend of people's sentiment in Twitter",
                           'showlegend':True}]),
               dict(label = 'Facebook',
                  method = 'update',
                  args = [{'visible': [False, False, True, True, False, False]},
                          {'title': "Month wise trend of people's sentiment in Facebook",
                           'showlegend':True}]),
                dict(label = 'WordCount',
                  method = 'update',
                  args = [{'visible': [False, False, False, False, True, True]},
                          {'title': "Month wise trend of Word count in Twitter and Facebook",
                           'showlegend':True}])
            ]),
            
        )
    ]
)

fig4.show()


# In[62]:


# Generate html file

pio.write_html(fig4, file='Social Media Trend.html', auto_open=True)


# In[63]:


# Kruskal-Wallis H-test

from scipy.stats import kruskal

# generate three independent samples
data1 = df_CT['PositiveSentiment']
data2 = df_Dv['PositiveSentiment']
data3 = df_LA['PositiveSentiment']
data4 = df_NY['PositiveSentiment']
data5 = df_Wa['PositiveSentiment']

# compare samples
stat, p = kruskal(data1, data2, data3, data4, data5)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')


# In[64]:


# Mann-Whitney U test

from scipy.stats import mannwhitneyu

# generate two independent samples
data1 = df_CT['PositiveSentiment']
data2 = df_Dv['PositiveSentiment']
data3 = df_LA['PositiveSentiment']
data4 = df_NY['PositiveSentiment']
data5 = df_Wa['PositiveSentiment']

# compare samples
stat, p = mannwhitneyu(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha: print('Same distribution (fail to reject H0)')
else: print('Different distribution (reject H0)')

stat, p = mannwhitneyu(data3, data4)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha: print('Same distribution (fail to reject H0)')
else: print('Different distribution (reject H0)')

stat, p = mannwhitneyu(data5, data1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha: print('Same distribution (fail to reject H0)')
else: print('Different distribution (reject H0)')


# In[ ]:




