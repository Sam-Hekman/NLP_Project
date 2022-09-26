# region imports
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

# endregion

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
maxReviews = 5000  # maximum number of reviews to get from a category

# file locations
trainReviewsFile = "datasetProcessing\\trainReviews.json"
testReviewsFile = "datasetProcessing\\testReviews.json"
trainTokenizedFile = "datasetProcessing\\trainTokenized.json"
testTokenizedFile = "datasetProcessing\\testTokenized.json"
trainStemmedFile = "datasetProcessing\\trainStemmed.json"
testStemmedFile = "datasetProcessing\\testStemmed.json"
trainLemmatizedFile = "datasetProcessing\\trainLemmatized.json"
testLemmatizedFile = "datasetProcessing\\testLemmatized.json"


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
                fileDownload = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/" + fileName
                urlretrieve(fileDownload, 'dataset\\' + fileName)  # download the file


# go through each json.gz and get the needed info
def ParseSplitDataset():
    trainingData = {}
    testData = {}
    for category in categories:
        reviews = []  # list of [reviewText, rating] lists for this category
        curReviews = 0  # counter to check we don't pass maxReviews
        file = category + "_5.json.gz"  # create file name
        content = gzip.open(os.path.join('dataset', file), 'r')  # get the content from the .json.gz file

        # go through each line (json) in the file
        for line in content:
            js = json.loads(line)  # convert line to json
            if "reviewText" in js:  # only keep reviews in dataset that have review text - excludes ones that are only ratings
                reviewInfo = [js["reviewText"].lower(), js["overall"]]  # get tuple (reviewText, rating)
                if reviewInfo not in reviews:  # only keep this review if it isn't a duplicate
                    reviews.append(reviewInfo)
                    curReviews += 1
            if curReviews == maxReviews:  # stop going through the file if we reach max reviews
                break

        splitPoint = int(curReviews * trainPercent)  # get the index at 70% of the data
        trainingData[category] = reviews[:splitPoint]  # training data should be first 70%
        testData[category] = reviews[splitPoint:]  # testing data should be remainder

    return trainingData, testData


# performs stemming on the dataset
def Stemming(getTraining: bool):
    # determine the dataset to stem
    if getTraining:
        file = trainTokenizedFile
    else:
        file = testTokenizedFile

    # get the tokenized reviews if we haven't already
    if getTraining and not os.path.exists(file):
        data, testing = ProcessData()
    elif not getTraining and not os.path.exists(file):
        training, data = ProcessData()
    else:  # if we have the tokenized text, open the file and load as a json
        reviewData = open(file)
        data = json.load(reviewData)

    stemmer = PorterStemmer()  # create nltk stemmer
    for category in data:  # for each category in the dataset
        for review in data[category]:  # for each review in the category
            review[0] = [stemmer.stem(word) for word in review[0]]  # stem each word in the review text

    return data


# performs lemmatization on the dataset
def Lemmatization(getTraining: bool):
    # determine the dataset to lemmatize
    if getTraining:
        file = trainLemmatizedFile
    else:
        file = testLemmatizedFile

    # get the tokenized reviews if we haven't already
    if getTraining and not os.path.exists(file):
        data, testing = ProcessData()
    elif not getTraining and not os.path.exists(file):
        training, data = ProcessData()
    else:  # if we have the tokenized text, open the file and load as a json
        reviewData = open(file)
        data = json.load(reviewData)

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

    return data


# function that runs through downloading, splitting up, and cleaning the dataset
# excludes stemming and lemmatization so these can be performed separately if needed
def ProcessData():
    ImportDataset()  # download json.gz files for dataset

    # Getting the data split into training and testing
    if os.path.exists(trainReviewsFile) and os.path.exists(testReviewsFile):
        reviewData = open(trainReviewsFile)
        train = json.load(reviewData)
        reviewData = open(testReviewsFile)
        test = json.load(reviewData)
    else:  # if it isn't already done, split the data and create files for it
        train, test = ParseSplitDataset()
        CreateJsonFile(trainReviewsFile, train)
        CreateJsonFile(testReviewsFile, test)

    # Getting the data with contractions expanded and text tokenized
    if os.path.exists(trainTokenizedFile) and os.path.exists(testTokenizedFile):
        reviewData = open(trainTokenizedFile)
        train = json.load(reviewData)
        reviewData = open(testTokenizedFile)
        test = json.load(reviewData)
    else:  # if not already done, run Normalize on the datasets
        train, test = Normalize(train, test)
        CreateJsonFile(trainTokenizedFile, train)
        CreateJsonFile(testTokenizedFile, test)

    return train, test


# function to remove contractions and tokenize the reviews
def Normalize(trainData, testData):
    # split up contractions and tokenize each review in the training data
    for category in trainData:
        for review in trainData[category]:
            noContractions = contractions.fix(review[0])  # contractions.fix separates into 2 words: aren't -> are not
            review[0] = word_tokenize(noContractions)  # word_tokenize splits sentences into words and punctuation

    # split up contractions and tokenize each review in the test data
    for category in testData:
        for review in testData[category]:
            noContractions = contractions.fix(review[0])
            review[0] = word_tokenize(noContractions)

    return trainData, testData


# endregion

# creates a json at given path using given data
def CreateJsonFile(filePath: str, data):
    # create directory if not present
    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):
        os.mkdir(directory)

    dataToJson = json.dumps(data)  # create json from data
    jsonFile = open(filePath, "w")  # create file
    jsonFile.write(dataToJson)  # write the json to the file
    jsonFile.close()  # close the file


# function to get the stemmed dataset
# getTraining -> if true, will return training dataset, if false will return testing dataset
def GetStemmedData(getTraining: bool):
    # determine which file to use
    if getTraining:
        file = trainStemmedFile
    else:
        file = testStemmedFile

    # if we already have the file, load it as json
    if os.path.exists(file):
        reviewData = open(file)
        data = json.load(reviewData)
    else:  # if we don't have the file, perform stemming and create the file
        data = Stemming(getTraining)
        CreateJsonFile(file, data)

    return data


# function to get the lemmatized dataset
# getTraining -> if true, will return training dataset, if false will return testing dataset
def GetLemmatizedData(getTraining: bool):
    # determine which file to use
    if getTraining:
        file = trainLemmatizedFile
    else:
        file = testLemmatizedFile

    # if we already have the file, load it as json
    if os.path.exists(file):
        reviewData = open(file)
        data = json.load(reviewData)
    else:  # if we don't have the file, perform lemmatization and create the file
        data = Lemmatization(getTraining)
        CreateJsonFile(file, data)

    return data