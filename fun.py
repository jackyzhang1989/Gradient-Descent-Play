
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path
import step1, step2, LR

data = pd.read_csv(str(Path.cwd().resolve()) + '/pandasDF.csv', dtype = {'User': str, 'ASIN': str, 'Rating': np.int})
print("Sample Data")
print("-----------")
print(data.sample(20))

# STEP1
# Count number of unique users and number of unique ASINs in our dataset
uniqueUsers, uniqueASINs, numUser, numASIN, train, test = step1.exe(data, train_test_split)


# STEP2
step2.exe(train, 'ASIN')
step2.exe(train, 'User')

# regression
countRatingsU, aveRatingsU, countRatingsA, aveRatingsA = LR.exe(train)
fig, axs = plt.subplots(ncols=2)
sns.regplot(x = countRatingsU, y = aveRatingsU, ax=axs[0])
sns.regplot(x = countRatingsA, y = aveRatingsA, ax=axs[1])
plt.show()

# In terms of parameters, the entire matrix P has number of films times number of users many variables. There are 1490
#  users in the dataset and 1186 books, meaning that P itself has 1,758,200 parameters, if we tried to specify all of them freely. 
# In a model we propose, each film has k features, and each user has k affinities, so that the total number of parameters in A and F
#  are 1490k+1186k=2670k. Thus, for instance if k=2 this is 5340 parameters, which is 0.3% of the total number of potential 
#  parameters there could be. This enormous compression is the first key component in our model.
# Build the matrices S and R from our training data

def computeMaskAndValMaze(S, R, data, uniqueUsers, uniqueASINs):
	for index, row in data.iterrows() :
		userIdx = uniqueUsers.index(row['User'])
		ASINIdx = uniqueASINs.index(row['ASIN'])
		S[userIdx, ASINIdx] = row['Rating']
		R[userIdx, ASINIdx] = 1
S = np.zeros((numUser,numASIN))
R = np.zeros((numUser,numASIN))
computeMaskAndValMaze(S, R, data, uniqueUsers, uniqueASINs)
K = R.sum()
    
# Build the matricies S and R from test data
Stest = np.zeros((numUser,numASIN))
Rtest = np.zeros((numUser,numASIN))
computeMaskAndValMaze(Stest, Rtest, test, uniqueUsers, uniqueASINs)    
Ktest = Rtest.sum()
print(K, Ktest, len(train), len(test))

# plot avg ASIN
averageRating = train['Rating'].mean()
filmRating = train.groupby(['ASIN'])['Rating'].mean()

aveVec = np.zeros(numASIN)
for i in range(0,numASIN) :
    aveVec[i] = filmRating[uniqueASINs[i]]

sns.distplot(aveVec)
plt.show()
#