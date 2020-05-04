# Rdd Caching & persist and self join concept

import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

def loadMovieNames():
    movieNames = {}
    with open('/home/sidhandsome/Coder_X/PySpark/ml-100k/u.item') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore')
    return movieNames

def makePairs((user, ratings)):
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, rating1), (movie2, rating2))

def filterDuplicates((userID, ratings)):
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2

def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_xy += ratingX * ratingY
        sum_yy += ratingY * ratingY
        numPairs += 1
    
    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)
    
    score = 0
    
    if denominator:
        score = (numerator/ float(denominator))
    
    return (score, numPairs)


conf = SparkConf().setMaster("local[*]").setAppName("PopularMovies")
sc = SparkContext(conf = conf)

data = sc.textFile('/home/sidhandsome/Coder_X/PySpark/ml-100k/u.data')

ratings = data.map(lambda x: x.split()).\
            map(lambda x: (int(x[0]), (int(x[1]), float(x[2]))))

joinedRatings = ratings.join(ratings)

uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

moviePairs = uniqueJoinedRatings.map(makepairs)

moviePairRatings = moviePairs.groupByKey()

moviePairSimilarities = moviePairRatings.map(computeCosineSimilarity).cache()

# Save the results 
#moviePairSimilarities.sortByKey()
#moviePairSimilarities.saveAsTextFile('movie-sims')

if (len(sys.argv) > 1):

    scoreThreshold = 0.97
    coOccurenceThreshold = 50

    movieID = int(sys.argv[1])

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = moviePairSimilarities.filter(lambda pairSim: \
        (pairSim[0][0] == movieID or pairSim[0][1] == movieID) \
        and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)

    # Sort by quality score.
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)

    print("Top 10 similar movies for " + nameDict[movieID])
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))

