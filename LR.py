def exe(train):
	# Count ratings and average ratings per User
	countRatingsU = train.groupby(['User'])['Rating'].count()
	aveRatingsU = train.groupby(['User'])['Rating'].mean()

	# Count ratings and average ratings per User
	countRatingsA = train.groupby(['ASIN'])['Rating'].count()
	aveRatingsA = train.groupby(['ASIN'])['Rating'].mean()
	return (countRatingsU, aveRatingsU, countRatingsA, aveRatingsA)



