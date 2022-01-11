from dotenv import load_dotenv
from ua_parser import user_agent_parser
import pickle
import os
import json
from datetime import datetime
import sqlite3
import pandas as pd
from statistics import mean
from fuzzywuzzy import fuzz
import math
import operator
import csv
import matplotlib.pyplot as plt
import matplotlib
from IPython import display
from base64 import b64decode
from PIL import Image
import io
import numpy as np
import base64
import random
from sklearn.metrics import jaccard_similarity_score
from difflib import SequenceMatcher
import time
import datetime
import uuid
from pyjarowinkler import distance

get_ipython().run_line_magic('matplotlib', 'inline')
load_dotenv(override=True)

# this can be manually done, assuming the user has access to the SQL database
# USER = os.getenv("USER")
# PASSWORD = os.getenv("PASSWORD")
# TABLE = os.getenv("TABLE")
# DATABASE = os.getenv("DATABASE")

# since the database is hosted locally on the same machine and the associated file is a
# sql file, the above step is not needed and you can connect directly by doing the following
# connecting to database
import sqlite3

# saved locally, no connection over network
conn = sqlite3.connect('../../bfp-pets-2020.sqlite')
cursor = conn.cursor()
print("Opened database successfully")
from IPython.core.pylabtools import figsize


# load all measurements in dataframe (2D datastructure given by pandas library)
measurements =  pd.read_sql_query("SELECT * FROM measurements", conn)


# the following function will receive a dataframe and an list of the needed attributes
# the function will then append the "id_measurement" and "measurement_datetime" (which are
# not entropy values for attributes so they will never appear naturally and must be manually
# appened so that they can be included in the dataset)

# After appending the values needed, the return value is the same dataframe without all the
# attributes not included in columns, also sorted in ascending order by the measurement_datetime
def createDataSet(measurements, columns = []):
    columns.append("id_measurement")
    columns.append("measurement_datetime")
    return measurements[columns].sort_values(by = 'measurement_datetime')


# Entropy values are imported from the csv file (entropy calculation done separetly in another notebook)
header_list = ["field", "entropy","normalizedEntropy"]
entropies = pd.read_csv('../allEntropiesNew.csv', names=header_list)

# Entropy values must not contain a field that is id_measurement, measurement_datetime, or id_user as those
# values for field give no information about the browsing environment with respect to its fingerprint
# furthermore, the stemmed values are also excluded
entropies = entropies[(entropies["field"] != "id_measurement") & (entropies["field"] != "measurement_datetime") & (entropies["field"] != "id_user") & (~entropies.field.str.contains("stemmed"))]

# entropy values are sorted in descending order by theit normalized entropy values
entropies = entropies.sort_values(by='normalizedEntropy', ascending=False)

# this function returns a list of N attributes with the highest entropy values
def getNBestAttributes(N):
    return entropies['field'].tolist()[:N]

# this function splits the dataframe contained in data into two chunks:
# 0 to numOrFloat, or if numOrFloat is a precentage from 0.0 to 1.0 then
# from 0 to size(dataset) * numOrFloat, and the other chunk being the rest
# of the data
def chronologicalSplit(numOrFloat, data):
    num = 37762

    if isinstance(numOrFloat, int):
        if numOrFloat < 0:
            num = 37762 - numOrFloat
        elif numOrFloat > 0:
            num = numOrFloat
    elif isinstance(numOrFloat, float):
        num = int(numOrFloat * num)

    return data.iloc[:num, :], data.iloc[num:, :]

# service function that returns the user_id from the fingerprint
def getUserIDFromRecordID(id, dataset):
    return dataset[dataset["id_measurement"] == id]["id_user"].values[0]

# service function that looks at the http_useragent to figure out whether the
# fingerprint associated with the id arguemnt is a mobile device or desktop.
def mobileOrDesktop(id):
    mobile_terms = ["Android", "iPhone", "iPad"]
    for mt in mobile_terms:
        if mt in measurements[measurements["id_measurement"] == id]["http_useragent"].values[0]:
            return "mobile"
    return "desktop"

# service function that gets a time in second and returns time in format
def human_time(secs):
    units = [("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)]
    parts = []
    for unit, mul in units:
        if secs / mul >= 1 or mul == 1:
            if mul > 1:
                n = int(math.floor(secs / mul))
                secs -= n * mul
            else:
                n = secs if secs != int(secs) else int(secs)
            parts.append("%s %s%s" % (n, unit, "" if n == 1 else "s"))

    return ", ".join(parts)


# the following block contains the code needed for the nodes of the linked list
# each node contains a fingerprint ID, a precentage match to the previous fingerprint
# a pointer/reference to the next fingerprint node.
class FpNode:
    def __init__(self, fpID, percentageMatchToPreviousFp = None):
        self.fpID = fpID
        self.next = None
        self.percentageMatchToPreviousFp = percentageMatchToPreviousFp

    def insertAfter(self, newFpNode):
        # if inserting at the end, overwrite the none with the reference to the new node
        if self.next == None:
            self.next = newFpNode

        # otherwise, store the pre-existing next in a temorary varaible, insert the new node
        # and restore the old next
        else:
            nextNode = self.next
            self.next = newFpNode
            newFpNode.next = nextNode

# the following code block contains the code for the linked list. Each linked list
# is associated with a browser environments and contains all the associated nodes
# a linked list has an associated ID (randomly generated), and a reference to the
# first and last fingerprint node in the list
class FpLinkedList:
    def __init__(self, generatedUserID,firstFp, latestFp):
        self.firstFp = firstFp
        self.generatedUserID = generatedUserID;
        self.latestFp = latestFp

    # getter
    def getUserID(self):
        return self.generatedUserID;

    # getter
    def getLatestFp(self):
        return self.latestFp

    # service function to print all the node IDs in the list
    def traverse_list(self):
        if self.firstFp is None:
            print("List has no element")
            return
        else:
            n = self.firstFp
            while n is not None:
                if n.next is not None:
                    print(n.fpID , "--> ", end = "")
                else:
                    print(n.fpID)
                n = n.next

    # Inserting node at end
    def insertAtEnd(self, newFpNode):
        if self.firstFp is None:
            self.firstFp = newFpNode
            self.lastFp = newFpNode
            return
        else:
            self.latestFp.next = newFpNode
            self.latestFp = newFpNode

    # suggested change, start from firstFp.next
    def printAccuracy(self,fp):
        if self.firstFp is None:
            return
        else:
            correctCounter = 0
            counter = 0
            n = self.firstFp.next
            prev = self.firstFp
            while n is not None:
                groundTruthN = getUserIDFromRecordID(n.fpID, dataset);
                groundTruthPrev = getUserIDFromRecordID(prev.fpID, dataset);
                if(groundTruthN == groundTruthPrev):
                    correctCounter += 1

                counter += 1
                prev = n
                n = n.next

            if(counter > 1):
                print((float(correctCounter/counter) * 100.0), "% accuracy")
                return

# returns true of the jaro_distance is above the threshold, false otherwise
# recieves as its arguments, two strings that hold the value of attributes
# in two fingerprints
def JaroWinklerSimilarity(val1,val2, threshold = 0.8, useExactValueForJaccardIndex = False):
    ratio = distance.get_jaro_distance(val1, val2, winkler=False, scaling=0.1)
    if(useExactValueForJaccardIndex):
        return ratio
    if(ratio > threshold):
        return True;
    else:
        return False;

# this function takes in an attribute name and two values for that attribute
# in two different fingerprints, and it returns true or false or the exact
# value for the JaroWinkler similarity score (depedning on the flag "useExactValueForJaccardIndex")
def matchAttributes(attributeName, value1, value2, useExactValueForJaccardIndex = False):
    # If the value being compared is one of the following: js_useragent,
    # http_useragent, or js_screen_resolution_avail_whcp, a specific
    # comparison threshold is used (or in the case of the last one, an exact match)
    # if any other attribute is used, a default of exact match is used.
    attributeToAlgorithmMap = {
        "js_useragent" : {
            "matchType" : "similarity",
            "threshold" : 0.8
        },
        "http_useragent": {
            "matchType" : "similarity",
            "threshold" : 0.8
        },
        "js_screen_resolution_avail_whcp" :{
            "matchType" : "exactEqual",
        },
        "default" : { # shouldn't this be similairty baed with threshold of 0.9?
            "matchType" : "exactEqual"
            # "threshold" : 0.9
        }
    }

    # uses the map to load into data
    data = attributeToAlgorithmMap["default"]
    if attributeName in attributeToAlgorithmMap:
        data = attributeToAlgorithmMap[attributeName]

    # depending on the value of data, either a specific match, or JaroWinkler similarity
    # algorithm is used
    if (data["matchType"] == "similarity"):
        return JaroWinklerSimilarity(value1, value2, data["threshold"], useExactValueForJaccardIndex)
    elif(data["matchType"] == "exactEqual"):
        return value1 == value2;

# wrapper function that handles the jaccard similarity index for all attributes in two
# fingerprints and returns a value between 0 and 1.0 that is the jaccard similarity index
def JaccardIndex(temp_id, fpLinkedList, D, useExactValueForJaccardIndex = False):
    # fetch the new (unmatched) fingerprint
    s1 = D[D['id_measurement'] == temp_id];
    # fetch the latest fingerprint in a certain browsing environment
    s2 = D[D['id_measurement'] == fpLinkedList.getLatestFp().fpID];

    # remove unnecssary columns
    s1 = s1.drop(columns=['id_measurement', 'measurement_datetime'])
    s2 = s2.drop(columns=['id_measurement', 'measurement_datetime'])

    # create a list of all attributes (should be 25)
    s1Columns = list(s1.columns)
    similaritySum = 0;

    for s1Col in s1Columns:
        # get the values in the two fingerprints
        s1ColValue = str(s1[s1Col].values[0])
        s2ColValue = str(s2[s1Col].values[0])

        # skip if either value is null
        if(s1ColValue == "" or s2ColValue == ""):
            continue;

        # run the matching algorithm for the two values
        runMatch = matchAttributes(s1Col, s1ColValue, s2ColValue, useExactValueForJaccardIndex);

        # if useExactValueForJaccardIndex is False then runMatch will just be a true or false value
        # as sich we just increment the similaritySum when True
        if(runMatch == True):
            similaritySum += 1

        # if useExactValueForJaccardIndex is True then we will get a float from matchAttributes that
        # we just add to the smiilary sum
        elif (type(runMatch) == float):
            similaritySum += runMatch

    totalNumAttributes = len(s1Columns)

    return (similaritySum / ((totalNumAttributes*2) - similaritySum))

# Iterates over the jaccard similarity scores for all possible fingerprints
# and returns the ID of the fingerprint with the highest Jaccard similarity index
# iff that similarity index is larger than the threshold, if not, None is returned
def JaccardMatchCriteria(scores, threshold = 0.8):
    currentMaxScore = 0
    currentMaxID = 0
    for fakeUserID in scores:
        if scores[fakeUserID] > currentMaxScore:
            currentMaxScore = scores[fakeUserID]
            currentMaxID = fakeUserID

    print("\t highest score found: ", currentMaxScore)
    if totalMatchesDone > 0:
        print("\t current acc = (", totalCorrectMatches, "/", totalMatchesDone, ")*100 = ", (totalCorrectMatches/totalMatchesDone)*100, "%")
    if(currentMaxScore > threshold):
        return {
            "topMatchUserID" : currentMaxID
        };
    else:
        return None

# recieves two fingerprints and returns True if the two browser fingerprints came from
# the same OS (or if either fingerprints does not have http_useragent)
def isFromSameOperatingSystem(temp_id, fpLinkedList, D):
    s1 = D[D['id_measurement'] == temp_id];
    s2 = D[D['id_measurement'] == fpLinkedList.getLatestFp().fpID];

    if(('http_useragent' not in s1) or ('http_useragent' not in s2)):
        return True
    else:
        osFamS1 = user_agent_parser.ParseOS(s1['http_useragent'].values[0])['family']
        osFamS2 = user_agent_parser.ParseOS(s2['http_useragent'].values[0])['family']

        if(osFamS1 == osFamS2):
            return True
        else:
            return False

# attack function, this is a dynamic attack since the fingerprints are given chronologically
# from oldest to newest
def attack(D, scoringFunction, matchCriteria, useExactValueForJaccardIndex = False, getMatchMetaData = False, threshold = 0.8, filePath = "test.txt" , startTime = time.time()):
    global totalMatchesDone
    global totalCorrectMatches

    X = {} # dict that maps user IDs (randomly generated) to a browsing environment linked list
    X_arr = [] # array that only contains user IDs (makes some stuff a bit easier)

    # this is the code that runs when a new browsing environment is found (which is guarnteed
    # to happen first time)

    if (len(X) == 0):
        firstRecord = D.iloc[[0]] # record is loaded from dataset
        newUserID = uuid.uuid1()  # random unique ID is generated
        fpNode = FpNode(int(firstRecord['id_measurement']))       # first node is created and passed to the node constructor
        newFpLinkedList = FpLinkedList(newUserID, fpNode, fpNode) # new linked list is created and the userID as well as the
                                                                  # node are passed note that the first and last Fp are both
                                                                  # fpNode as this is a new list with only one node

        X[newUserID] = newFpLinkedList # new browsing environment is added to the dictionary
        X_arr.append(newUserID)        # new user ID is added to the array

    index = 1;
    globalTemp = {} # unsure of what this does
    matchMetaData = [] # only needed if getMatchMetaData is True but keep it

    # keep iterating while you still have data in the dataset
    while(index != len(D)):
        print((index / len(D))* 100, "% complete")
        temp = D.iloc[[index]]; # temp has the fingerprint being considered/processed
        tempScores = {} # tempScores is the dictionary that holds as its key, the browsing environments ID / user IDs
                        # and as its value, the similarity score between the current fingerprint and the latest fp
                        # of each browsing environment or linked list

        temp_id = int(temp['id_measurement']) # this is the id of said fingerprint, used to find it in the dataset

        # the above value is used to identify the current fingerprint, so:
            # D[temp_id] or X[temp_id] or dataset[temp_id] will return the current fingerprint
            # whilst X[fakeUserID] shown below will return the linked list that contains the
            # browsing environment associated with that ID

        # iterate over all user IDs that are known
        for fakeUserID in reversed(X_arr):
            # if the two fingerprints are not from the same OS they are definitely not from the same browser
            # set score to zero and continue
            if not isFromSameOperatingSystem(temp_id, X[fakeUserID], D):
                tempScores[fakeUserID] = 0;
                continue;

            # add the score of each browsing environment
            tempScores[fakeUserID] = scoringFunction(temp_id, X[fakeUserID], D, useExactValueForJaccardIndex) # returns score for specific fake userID

            # if metaData is required, add it
            if getMatchMetaData:
                bID = X[fakeUserID].getLatestFp().fpID
                trueMatch = getUserIDFromRecordID(temp_id, D) == getUserIDFromRecordID(bID, D)
                matchMetaData.append({
                   "fpA_id" :  temp_id,
                    "fpB_id" : bID,
                    "score" : tempScores[fakeUserID],
                    "groundTruthMatch" : trueMatch,
                    "fpA_device" : mobileOrDesktop(temp_id),
                    "fpB_device" : mobileOrDesktop(bID),
                });

            # if we get a perfect match of 1, we break early and skip all subsequent fingerprints
            if tempScores[fakeUserID] == 1:
                break;

        # we iterate over all scores and return the ID of the highest matching browsing environment
        runMatchCriteria = matchCriteria(tempScores, threshold) #returns keys topMatchUserID


        # printing progress status
        if(index % 100) == 0:
            appendString = "\nIteration: " + str(index);
            if totalMatchesDone > 0:
                perc = (totalCorrectMatches/totalMatchesDone)*100;
                appendString += "\n\t acc = (" + str(totalCorrectMatches) + "/" + str(totalMatchesDone) + ") * 100 = " + str(perc) + "%";
            appendString += "\n\t no. of browsing env = " + str(len(X_arr))
            #appendString += "\n\t total matches computed: " + str(len(matchMetaData))
            curTime = time.time()
            appendString += "\n\t time taken so far: "+ human_time(curTime - startTime)
            appendString += "\n\t (" + str(index) + "/" + str(len(D)) + ") * 100 = " + str((index / len(D))* 100) + " % complete"
            appendString += "\n--------------------------------------------"

            with open(filePath, 'a+') as file:
                file.seek(0)
                content = file.read()
                if appendString not in content:
                    file.write(appendString)

        # if runMatchCriteria is not None, meaning the matchCriteria function returned an ID
        # we instantiate a new node, and append this new node to the corresponding linked list
        if(runMatchCriteria != None):
            totalMatchesDone += 1
            newFpNode = FpNode(temp_id, tempScores[fakeUserID])

            # we first fetch the old fingerprint ID
            oldFpID = X[runMatchCriteria['topMatchUserID']].getLatestFp().fpID
            #append the new one
            X[runMatchCriteria['topMatchUserID']].insertAtEnd(newFpNode)

            # compare the user ID of both to see if the match is correct
            groundTruthS1 = getUserIDFromRecordID(temp_id, dataset)
            groundTruthS2 = getUserIDFromRecordID(oldFpID, dataset)
            print("\t\ttemp_id",temp_id,"oldFpID",oldFpID)
            print("\t\tgroundTruthS1",groundTruthS1,"groundTruthS2",groundTruthS2)

            # if match is correct, increment the correct matches counter
            if groundTruthS1 == groundTruthS2:
                totalCorrectMatches += 1

        # if matchCriteria did return None, then this a is a new browsing environment so the
        # program does what it did initially, which is make a new instance of a browser fingerprint
        else:
            newFakeUserID = uuid.uuid1()
            fpNode = FpNode(temp_id)
            newFpLinkedList = FpLinkedList(newFakeUserID, fpNode, fpNode)
            X[newFakeUserID] = newFpLinkedList
            X_arr.append(newFakeUserID)

        index += 1


    return X , matchMetaData

# this is a wrapper function used to run the attack.
def testAttack(size, attr_num = 25, threshold = 0.6):

    startTime = time.time()

    global totalMatchesDone = 0
    global totalCorrectMatches = 0

    dataset, d2 = chronologicalSplit(size,createDataSet(measurements,getNBestAttributes(attr_num)))
    datasetGroundTruthed , d3 = chronologicalSplit(size,createDataSet(measurements,['id_user']))

    groundTruth_number_of_users = datasetGroundTruthed['id_user'].nunique()

    path = "attack7_results/results_"+ str(attr_num) + "attrs_" + str(threshold) +"threshold_" + str(size) +"size.txt"

    attackResults, metaData = attack(dataset, JaccardIndex, JaccardMatchCriteria, False, False, threshold, path, startTime)

    endTime = time.time()

    percentageAcc = (totalCorrectMatches/totalMatchesDone)*100;

    appendString = "\nThreshold: " + str(threshold) + " | Records: " + str(size) + " | Top " + str(attr_num) + " Attributes";
    appendString += "\n\t acc = (" + str(totalCorrectMatches) + "/" + str(totalMatchesDone) + ") * 100 = " + str(percentageAcc) + "%";
    appendString += "\n\t total matches computed: " + str(len(metaData))
    appendString += "\n\t number of users in ground truth: " + str(groundTruth_number_of_users)
    appendString += "\n\t number of browsing environments found: " + str(len(attackResults))
    appendString +=  "\n\tTotal time taken: " + human_time(endTime - startTime)

    path = "attack7_results/finalResults_"+ str(attr_num) + "attrs_" + str(threshold) +"threshold_" + str(size) +"size.txt"

    with open(path, 'a+') as file:
        file.seek(0)
        content = file.read()
        if appendString not in content:
            file.write(appendString)

    print(appendString)

testAttack(37762, 25, 0.9)
