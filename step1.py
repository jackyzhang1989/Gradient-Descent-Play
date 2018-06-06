def exe(data, train_test_split):
	# STEP1
	# Count number of unique users and number of unique ASINs in our dataset
	uniqueUsers = data['User'].unique().tolist()
	uniqueASINs = data['ASIN'].unique().tolist()
	numUser = len(uniqueUsers)
	numASIN = len(uniqueASINs)

	# Split to train and test
	train, test = train_test_split(data, random_state = 8675309, stratify = data['ASIN'])
	num_train = train.shape[0]
	num_test = test.shape[0]

	print("Number of Users: {}".format(numUser))
	print("Number of ASINs: {}".format(numASIN))
	return (uniqueUsers, uniqueASINs, numUser, numASIN, train, test)


