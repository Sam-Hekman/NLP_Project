import os
import json
import gzip
import contractions
from collections import defaultdict
from urllib.request import urlretrieve
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# region variables
categories = ["AMAZON_FASHION", "All_Beauty", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Books",
              "CDs_and_Vinyl",
              "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics", "Gift_Cards",
              "Grocery_and_Gourmet_Food", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
              "Luxury_Beauty",
              "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products",
              "Patio_Lawn_and_Garden",
              "Pet_Supplies", "Prime_Pantry", "Software", "Sports_and_Outdoors", "Tools_and_Home_Improvement",
              "Toys_and_Games",
              "Video_Games"]  # the categories in the Amazon review dataset
trainPercent = .7  # percentage of data to use for training
maxReviews = 1700  # maximum number of reviews to get from a category
# Leave at none to use the global random state instance from numpy.random, use an integer for a reproducible output
splitRandomSeed = 1
negativeCutoff = 2.0
neutralCutoff = 4.0
positiveCutoff = 5.0

# file locations
reviewsFile = "datasetProcessing\\reviews.json.gz"
tokenizedCleanedFile = "datasetProcessing\\tokenizedCleaned.json.gz"
stemmedFile = "datasetProcessing\\stemmed.json.gz"
lemmatizedFile = "datasetProcessing\\lemmatized.json.gz"
labels = ["positive", "neutral", "negative"]


# endregion

# region getting and normalizing data
# downloads needed json.gz files from https://nijianmo.github.io/amazon/index.html#code
def ImportDataset():
    # create folder if not present
    if not os.path.exists('dataset'):
        os.mkdir('dataset')

    # if we don't have all of the needed files downloaded
    if len(os.listdir('dataset')) is not len(categories):
        for category in categories:  # go through all of the categories
            fileName = category + "_5.json.gz"
            if not os.path.exists('dataset\\' + fileName):  # verify we don't already have the file
                print(fileName + " downloading")
                # fileDownload = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/" + fileName
                fileDownload = "http://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/" + fileName
                urlretrieve(fileDownload, 'dataset\\' + fileName)  # download the file


# go through each json.gz and get the needed info
def GetDataset():
    dataset = {}
    for category in categories:
        reviews = []  # list of [reviewText, rating] lists for this category
        curNegativeReviews = 0  # counter to check we don't pass maxReviews
        curNeutralReviews = 0
        curPositiveReviews = 0
        curReviews = 0
        file = category + "_5.json.gz"  # create file name
        content = gzip.open(os.path.join('dataset', file), 'r')  # get the content from the .json.gz file

        # go through each line (json) in the file
        for line in content:
            js = json.loads(line)  # convert line to json
            if "reviewText" in js:  # only keep reviews in dataset that have review text - excludes ones that are only ratings
                reviewInfo = [js["reviewText"].lower(), js["overall"]]  # get tuple (reviewText, rating)
                score = reviewInfo[1]
                if reviewInfo not in reviews:  # only keep this review if it isn't a duplicate
                    # check which category the review falls in and only keep it if that class hasn't exceeded its limit
                    if score <= negativeCutoff and curNegativeReviews < maxReviews:
                        curNegativeReviews += 1
                        reviews.append(reviewInfo)
                        curReviews += 1
                    elif score <= neutralCutoff and curNeutralReviews < maxReviews:
                        curNeutralReviews += 1
                        reviews.append(reviewInfo)
                        curReviews += 1
                    elif score <= positiveCutoff and curPositiveReviews < maxReviews:
                        curPositiveReviews += 1
                        reviews.append(reviewInfo)
                        curReviews += 1
            if curReviews == maxReviews * 3:  # stop going through the file if we reach max reviews
                break

        dataset[category] = reviews
    return dataset


# function to get the stemmed dataset
def PerformStemming():
    # if we already have the file, load it as json and return
    if os.path.exists(stemmedFile):
        with gzip.open(stemmedFile, 'r') as fin:
            data = json.loads(fin.read().decode('utf-8'))
        return data

    data = ProcessData()  # get the tokenized and cleaned reviews
    stemmer = PorterStemmer()  # create nltk stemmer
    for category in data:  # for each category in the dataset
        for review in data[category]:  # for each review in the category
            review[0] = [stemmer.stem(word) for word in review[0]]  # stem each word in the review text

    CreateJsonFile(stemmedFile, data)  # create the file with stemmed data
    return data


# function to get the lemmatized dataset
def PerformLemmatization():
    # if we already have the file, load it as json and return
    if os.path.exists(lemmatizedFile):
        with gzip.open(lemmatizedFile, 'r') as fin:
            data = json.loads(fin.read().decode('utf-8'))
        return data

    data = ProcessData()  # get the tokenized and cleaned reviews
    lemmatizer = WordNetLemmatizer()  # create nltk lemmatizer
    # create mapping of tags -> needed to give part of speech of the tokens to the lemmatizer
    tagMap = defaultdict(lambda: wordnet.NOUN)
    tagMap['J'] = wordnet.ADJ
    tagMap['V'] = wordnet.VERB
    tagMap['R'] = wordnet.ADV

    for category in data:  # for each category in the dataset
        for review in data[category]:  # for each review in the category
            # for each token and pos tag, lemmatize the token, giving the needed tag
            review[0] = [lemmatizer.lemmatize(word, tagMap[tag[0]]) for word, tag in pos_tag(review[0])]

    CreateJsonFile(lemmatizedFile, data)
    return data


# function that runs through downloading, splitting up, and cleaning the dataset
# excludes stemming and lemmatization so these can be performed separately if needed
def ProcessData():
    # Getting the data cut down so full set isn't used
    if os.path.exists(reviewsFile):
        with gzip.open(reviewsFile, 'r') as fin:
            data = json.loads(fin.read().decode('utf-8'))
    else:  # if it isn't already done, split the data and create files for it
        ImportDataset()  # download json.gz files for dataset
        data = GetDataset()
        CreateJsonFile(reviewsFile, data)

    # Getting the data with contractions expanded and text tokenized
    if os.path.exists(tokenizedCleanedFile):
        with gzip.open(tokenizedCleanedFile, 'r') as fin:
            data = json.loads(fin.read().decode('utf-8'))
    else:  # if not already done, run Normalize on the datasets
        data = Normalize(data)
        CreateJsonFile(tokenizedCleanedFile, data)

    return data


# function to remove contractions and tokenize the reviews
def Normalize(data):
    # split up contractions and tokenize each review in the training data
    for category in data:
        for review in data[category]:
            noContractions = contractions.fix(review[0])  # contractions.fix separates into 2 words: aren't -> are not
            review[0] = word_tokenize(noContractions)  # word_tokenize splits sentences into words and punctuation

    return data


# endregion

# creates a json at given path using given data
def CreateJsonFile(filePath: str, data):
    # create directory if not present
    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):
        os.mkdir(directory)

    dataToJson = json.dumps(data)  # create json from data
    jsonBytes = dataToJson.encode('utf-8')
    with gzip.open(filePath, 'w') as fout:  # 4. fewer bytes (i.e. gzip)
        fout.write(jsonBytes)


def GetDatasetStats(training, testing):
    print("-------- Stats --------")
    tokenCountTrain = 0
    posCountTrain = 0
    neutralCountTrain = 0
    negCountTrain = 0
    tokenCountTest = 0
    posCountTest = 0
    neutralCountTest = 0
    negCountTest = 0
    for category in training:
        for reviews in training[category]:
            tokenCountTrain += len(reviews[0])
            if reviews[1] < negativeCutoff:
                negCountTrain += 1
            elif reviews[1] < neutralCutoff:
                neutralCountTrain += 1
            elif reviews[1] < positiveCutoff:
                posCountTrain += 1
        for reviews in testing[category]:
            tokenCountTest += len(reviews[0])
            if reviews[1] < negativeCutoff:
                negCountTest += 1
            elif reviews[1] < neutralCutoff:
                neutralCountTest += 1
            elif reviews[1] < positiveCutoff:
                posCountTest += 1

    print("Training data: ")
    print("\tTokens: {0}".format(tokenCountTrain))
    print("\tNegative Reviews: {0}".format(negCountTrain))
    print("\tNeutral Reviews: {0}".format(neutralCountTrain))
    print("\tPositive Reviews: {0}".format(posCountTrain))
    print("\nTest data: ")
    print("\tTokens: {0}".format(tokenCountTest))
    print("\tNegative Reviews: {0}".format(negCountTest))
    print("\tNeutral Reviews: {0}".format(neutralCountTest))
    print("\tPositive Reviews: {0}".format(posCountTest))


# combine all data, then split into training and testing
# if stem is True, returns stemmed data, otherwise returns lemmatized
# returns train data, test data
def SplitData(stem: bool):
    # choose which file for stemming or lemmatization
    if stem:
        data = PerformStemming()
    else:
        data = PerformLemmatization()

    allData = sum(data.values(), [])  # combine all of the data
    # split the data, giving the trainPercent and random seed
    split = train_test_split(allData, train_size=trainPercent, random_state=splitRandomSeed)

    GetDatasetStats(split[0], split[1])
    return split[0], split[1]  # return train data, test data


# split data into training and testing while still in categories
# if stem is True, returns stemmed data, otherwise returns lemmatized
# returns train data, test data
def SplitDataKeepCategories(stem: bool):
    # choose which file for stemming or lemmatization
    if stem:
        data = PerformStemming()
    else:
        data = PerformLemmatization()

    train = {}
    test = {}

    for category in data:
        # split the data, giving the trainPercent and random seed
        split = train_test_split(data[category], train_size=trainPercent, random_state=splitRandomSeed)
        train[category] = split[0]  # update training and testing dictionary for this category
        test[category] = split[1]

    GetDatasetStats(train, test)
    return train, test


# returns precision, recall, and F1 score (in order) for each class in the matrix as a dictionary
def GetScores(y_true, y_pred):
    stats = {}  # dictionary to hold the calculations for each class

    # getting precision, recall, and f1 for all classes
    precision = precision_score(y_true, y_pred, average=None, labels=labels)
    recall = recall_score(y_true, y_pred, average=None, labels=labels)
    f1 = f1_score(y_true, y_pred, average=None, labels=labels)

    i = 0
    for l in labels:  # loop through each class
        stats[l] = [precision[i], recall[i], f1[i]]  # get the info for this class specifically
        i += 1

    return stats  # return the dictionary


# testing for confusion matrix - getting F1 score
# true = ["positive", "neutral", "positive", "positive", "neutral", "negative"]
# pred = ["neutral", "neutral", "positive", "positive", "neutral", "positive"]
# print(GetFScores(true, pred))

SplitDataKeepCategories(True)
