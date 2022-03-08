#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
from surprise import Dataset
from surprise import Reader
import matplotlib.pyplot as plt
from surprise import SVDpp
from surprise import SVD
from surprise import NMF
from surprise import KNNBasic
from surprise import CoClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy
import gzip
import math
sb.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ratings = pd.read_csv('Data/book_ratings.csv')
ratings.head()


# In[3]:


books = pd.read_csv('Data/item_info.csv')
books = books[['item_id','ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.head()


# In[4]:


Data = pd.merge(ratings, books, on='item_id')
Data.head()


# In[5]:


print('# of items : ', Data.item_id.count())
print('# of user : ', Data.user.count())
print('Mean of Ratings : ', Data.rating.mean())


# In[6]:


Data.groupby('Book-Title')['rating'].mean().sort_values(ascending = False).head()


# In[7]:


Data.groupby('Book-Title')['rating'].count().sort_values(ascending = False).head(10)


# In[8]:


rs = pd.DataFrame(Data.groupby('Book-Title')['rating'].mean())
rs['num_ratings'] = Data.groupby('Book-Title')['rating'].count()
plt.figure(figsize=(10,5))
rs.num_ratings.hist(bins=70)
plt.show()


# In[9]:


plt.figure(figsize=(10,5))
rs.rating.hist(bins=70)
plt.show()


# In[10]:


d = Data[['item_id', 'user', 'rating']]
reader = Reader(rating_scale = (1,10))
data = Dataset.load_from_df(d, reader)
train , test = train_test_split(data,test_size = 0.2)


# In[11]:


algorithm1 = SVDpp(n_factors = 20, n_epochs= 20,lr_all=0.01,init_std_dev= 0.005,verbose=False, reg_all=0.01)
algorithm2 = SVD(n_factors = 20, n_epochs= 20,lr_all=0.01,init_std_dev= 0.005,verbose=False, reg_all=0.01)
algorithm3 = KNNBasic()
algorithm1.fit(train)
algorithm2.fit(train)
algorithm3.fit(train)
pred1 = algorithm1.test(test)
pred2 = algorithm2.test(test)
pred3 = algorithm3.test(test)
accuracy.rmse(pred1)
accuracy.rmse(pred2)
accuracy.rmse(pred3)
accuracy.mae(pred1)
accuracy.mae(pred2)
accuracy.mae(pred3)


# In[12]:


real_rating=[]
pred_rating=[]
real_rating2=[]
pred_rating2=[]
real_rating3=[]
pred_rating3=[]
for i in pred1:
    pui = pred1.pop()
    real_rating.append(pui.r_ui)
    pred_rating.append(pui.est)
for i in pred2:
    pui2 = pred2.pop()
    real_rating2.append(pui2.r_ui)
    pred_rating2.append(pui2.est)
for i in pred3:
    pui3 = pred3.pop()
    real_rating3.append(pui3.r_ui)
    pred_rating3.append(pui3.est)


# In[13]:


ratings_frame = pd.DataFrame({'Real_rating':real_rating,'Predicted_rating':pred_rating})
ratings_frame3 = pd.DataFrame({'Real_rating':real_rating3,'Predicted_rating':pred_rating3})
error = (abs((ratings_frame['Real_rating'] - ratings_frame['Predicted_rating'])) / ratings_frame['Real_rating'])*1e2
error3 = (abs((ratings_frame3['Real_rating'] - ratings_frame3['Predicted_rating'])) / ratings_frame3['Real_rating'])*1e2
fig, axes = plt.subplots(1,2,figsize=(20,5))
axes[0].hist(error,bins=[i/100 for i in range(3,100)])
axes[1].hist(error3,bins=[i/100 for i in range(3,100)], color = 'r')
plt.show()
fig.savefig('Book-Crossing.jpg', dpi=900)


# In[14]:


d = pd.DataFrame(pred1, columns = ['user_id','item_id','real_ui','estimated_value','details'])
ranking = d.groupby('item_id')['estimated_value'].count().sort_values(ascending = False)
item_id = ranking.index
Data_to_np = Data.to_numpy()
Book_titles = []
for i in range(0,len(item_id)):
    Data_to_np[i,1] == item_id[i]
    Book_titles.append(Data_to_np[i,4])

Book_titles = list(set(Book_titles))
final = np.zeros((48,2), dtype = object)
for i in range(0, len(final)):
    final[i,0] = Book_titles[i]
    final[i,1] = item_id[i]
final = pd.DataFrame(final, columns=['Title', 'ID'])
final


# In[15]:


def parse(path):
    g = gzip.open(path, 'r')
    false = False
    true = True
    for l in g:
        yield eval (l)
def gen_toNumpy(gen):
    data = np.zeros((100000,5),dtype = object)
    for i in range(0,len(data)):
        for j in range(0,len(data[0])):
            r = next(gen)
            d = np.array(list(r.values()))
            data[i,j] = d[j]
    return data
good_reads = parse('Data/goodreads.gz')
d = gen_toNumpy(good_reads)


# In[16]:


d[:,4] = d[:,4].astype(np.int)
DataCsv = np.zeros((len(d),len(d[0])), dtype = object)
for i in range(0,len(DataCsv)):
    if d[i,4] != 0:
        for j in range(0,len(DataCsv[0])):
            DataCsv[i,j] = d[i,j]
DataCsv = DataCsv[~np.all(DataCsv==0, axis=1)]


# In[17]:


dt = pd.read_csv('Data/goodreads.csv')
dt = dt[['item_id', 'user', 'rating']]
reader = Reader(rating_scale=(1,5))
dt = Dataset.load_from_df(dt,reader)
train, test = train_test_split(dt, test_size=0.2)


# In[18]:


algorithm1 = SVDpp(n_factors = 20, n_epochs= 20,lr_all=0.01,init_std_dev= 0.005,verbose=False, reg_all=0.01)
algorithm2 = SVD(n_factors = 20, n_epochs= 20,lr_all=0.01,init_std_dev= 0.005,verbose=False, reg_all=0.01)
algorithm3 = KNNBasic()
algorithm1.fit(train)
algorithm2.fit(train)
algorithm3.fit(train)
pred1 = algorithm1.test(test)
pred2 = algorithm2.test(test)
pred3 = algorithm3.test(test)
accuracy.rmse(pred1)
accuracy.rmse(pred2)
accuracy.rmse(pred3)
accuracy.mae(pred1)
accuracy.mae(pred2)
accuracy.mae(pred3)


# In[19]:


real_rating=[]
pred_rating=[]
real_rating2=[]
pred_rating2=[]
real_rating3=[]
pred_rating3=[]
for i in pred1:
    pui = pred1.pop()
    real_rating.append(pui.r_ui)
    pred_rating.append(pui.est)
for i in pred2:
    pui2 = pred2.pop()
    real_rating2.append(pui2.r_ui)
    pred_rating2.append(pui2.est)
for i in pred3:
    pui3 = pred3.pop()
    real_rating3.append(pui3.r_ui)
    pred_rating3.append(pui3.est)


# In[20]:


ratings_frame = pd.DataFrame({'Real_rating':real_rating,'Predicted_rating':pred_rating})
ratings_frame3 = pd.DataFrame({'Real_rating':real_rating3,'Predicted_rating':pred_rating3})
error = (abs((ratings_frame['Real_rating'] - ratings_frame['Predicted_rating'])) / ratings_frame['Real_rating'])*1e2
error3 = (abs((ratings_frame3['Real_rating'] - ratings_frame3['Predicted_rating'])) / ratings_frame3['Real_rating'])*1e2
fig, axes = plt.subplots(1,2,figsize=(20,5))
axes[0].hist(error,bins=[i/100 for i in range(0,100)], color = 'g')
axes[1].hist(error3,bins=[i/100 for i in range(0,100)], color = 'k')
plt.show()
fig.savefig('Goodreads.jpg', dpi=900)


# In[ ]:




