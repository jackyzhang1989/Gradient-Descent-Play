def exe(train, index):
	# Count ratings per ASIN
	countRatings = train.groupby([index])['Rating'].count()

	# Count numberer of ASINs in train
	numASIN_train = float(countRatings.shape[0])

	# Count ratings per user above 32 and 64 and 128
	num32 = countRatings[countRatings > 32].count()
	num64 = countRatings[countRatings > 64].count()
	num128 = countRatings[countRatings > 128].count()

	# Print the fraction
	print("Fraction of ASINs > 32 Ratings: {}".format(num32/numASIN_train))
	print("Fraction of ASINs > 64 Ratings: {}".format(num64/numASIN_train))
	print("Fraction of ASINs >128 Ratings: {}".format(num128/numASIN_train))


