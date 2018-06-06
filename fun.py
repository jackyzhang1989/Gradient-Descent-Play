
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path
import step1, step2

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


