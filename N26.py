import time
import pandas as pd
import numpy as np
from sklearn import tree, preprocessing, svm
from sklearn.cross_validation import train_test_split, cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pickle

####Test instructions
# In order to test on original dataset:
# 1. Place N25_Train.csv in same directory as script
# 2. Run
# In order to test script on new dataset:
# 1. Make sure rf_model.p file in same directory - this contains pre-trained model from original training set
# 2. Change test_csv_path to new filename
# 3. Set use_trained_model to True
# 4. Run

test_csv_name = "N26_TRAIN.csv" # change filename for new test file

use_trained_model = False
# if False, script will train on .csv file given and produce new model file
# if True, script will use pre-trained model from original dataset and display classification accuracy

######### PART I - User split ####

df = pd.read_csv(test_csv_name)
### Dataframe Preliminary Steps
# timestamp conversions from epoch to year/month/etc :
get_year = lambda epoch: time.gmtime(epoch/1000)[0]
def get_formatted_month(epoch):
    year = time.gmtime(epoch/1000)[0]
    month = str(time.gmtime(epoch/1000)[1])
    if len(month) == 1:
        month = "0" + month
    date_string = "{0}-{1}".format(year, month)
    return date_string

# future function: see if signup was done at night time (when no verification service available)
get_weekday = lambda epoch: time.gmtime(epoch/1000)[6]  # Monday = 0...Sunday = 6

# time.struct format for indexing: time.struct_time(tm_year=2015, tm_mon=9, tm_mday=25, tm_hour=15,
# tm_min=29, tm_sec=42, tm_wday=4, tm_yday=268, tm_isdst=0)

# clean up each row:
# Reformat "birthdate" as birth year (assume individual birthmonth/day won't matter for prediction)
df["birthdate"] = df["birthdate"].apply(get_year)

# introduce new variable: weekday during which initial signup was created
df["created_weekday"] = df["created"].apply(get_weekday)

# Reformat "created"-timestamp as month and year joined
# for use for plotting monthly trends
df["joined_year"] = df["created"].apply(get_formatted_month)

# calculate days since first customer joined
# (where start time is either min. starting date value, or 1.1.2013 in case there's a 0-date)
# for use in learning model
df["time_since_business_start"] = (df["created"]- max(df["created"].min(), 1356998400))/(1000*3600*24)

# Replace updated-time with time difference between updates
# I'm guessing very short or very long time differences indicate trouble
df["timedelta"] = (df["updated"]-df["created"])/(1000*3600*24) # add timedelta in days

# delete time variables that we have replaced
del df["created"], df["updated"]

# assuming ID, last name, birthplace not helpful during learning process (without advanced processing):

del df["id"], df["lastName"], df["birthPlace"]

# leave gender variable as is

# deal with nationality in next sections - will sort into 4 largest bins

### Plots ###

# split up by age - dropping people > 90 years of age
birthyear_split = df[df["birthdate"]>1925].groupby("birthdate")

# split up by passport type
passport_split = df.groupby("passportType")
# get some summary statistics in terms of how often each one "fails"

# split up by weekday created
weekday_split = df.groupby("created_weekday")

# split up by joining year
time_joined_split = df.groupby("joined_year")

# split up by birth country
counter = df.groupby("nationality")["signupCompleted"].count().sort_values(ascending=False)
sums = df.groupby("nationality")["signupCompleted"].sum()[counter.index]
max_countries_shown = 20
ind = np.arange(max_countries_shown)
unsuccessful = counter - sums
p1 = plt.bar(ind, sums[0:20], color='b')
p2 = plt.bar(ind, unsuccessful[0:20], bottom=sums[0:20], color='r')
plt.xticks(ind+0.35,counter.index[0:20])
plt.title("Top 20 Nationalities")

# For subplot, easier to visualise under largest groups, i.e. German, Austrian, EU, Non-EU
EU_COUNTRY_CODES = ["AUT","BEL","BGR","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC","HUN","IRL","ITA",
                    "LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK","SVN",
                    "ESP","SWE","GBR"]

# use second set here to not lose track of changes to original dataframe
df_nationalised = df
def nationaliser(country):
    # sort country names into Germany/Austria/non-german EU/ non-EU states
    if country not in ("AUT", "DEU"):
        if country in EU_COUNTRY_CODES:
            return "other EU"
        else:
            return "non-EU"
    else:
        return country
df_nationalised["nationality"] = df_nationalised["nationality"].apply(nationaliser)
nationality_split = df_nationalised.groupby("nationality")

f, axarr = plt.subplots(2, 2)

# subplot: nationality
# sort by headcount, since we don't have to keep natural order for Nationality (unlike birthyear, year joined, etc).
counter = nationality_split["signupCompleted"].count().sort_values(ascending=False)
sums = nationality_split["signupCompleted"].sum()[counter.index]
ind = np.arange(len(counter.index))
# group by nationality - sort by
#plt.xticks(ind + 0.35, counter.index)
plt.sca(axarr[0, 0])
plt.xticks(ind + 0.35, counter.index)
axarr[0, 0].bar(ind, sums, color='b')
axarr[0, 0].bar(ind, counter - sums, bottom=sums, color='r')
axarr[0, 0].set_title('Nationality Groups')

# subplot: birthyear
counter = birthyear_split["signupCompleted"].count()  # how many users per group
sums = birthyear_split["signupCompleted"].sum()  # how many completed signups
ind = np.arange(len(counter.index))
plt.sca(axarr[0, 1])
plt.xticks(ind[::5], counter.index[::5], rotation=60)
axarr[0, 1].bar(ind, sums, color='b')
axarr[0, 1].bar(ind, counter - sums, bottom=sums, color='r')
axarr[0, 1].set_title('Birthyear Distribution')

# subplot: weekday
counter = weekday_split["signupCompleted"].count()  # how many users per group
sums = weekday_split["signupCompleted"].sum()  # how many completed signups
ind = np.arange(len(counter.index))
plt.sca(axarr[1, 0])
plt.xticks(ind + 0.35, ("Mon", "Tue", "Wed", "Thu", "Fr","Sa","So"))
axarr[1, 0].bar(ind, sums, color='b')
axarr[1, 0].bar(ind, counter - sums, bottom=sums, color='r')
axarr[1, 0].set_title('Weekday Joined')

# subplot: joining time
counter = time_joined_split["signupCompleted"].count()
sums = time_joined_split["signupCompleted"].sum()
ind = np.arange(len(counter.index))
plt.sca(axarr[1, 1])
plt.xticks(ind, counter.index, rotation=60)
axarr[1, 1].bar(ind, sums, color='b')
axarr[1, 1].bar(ind, counter - sums, bottom=sums, color='r')
axarr[1, 1].set_title('Month Joined')
plt.show()

# investigate last month 2015 separately:
# df[df["joined_year"]=="2015-12"].to_csv("why_so_many_failures.csv")

###### Part II  ######
# sort nationalities into 4 bins, I don't want to overtrain on rare, single-mention countries
df["nationality"] = df["nationality"].apply(nationaliser)

# joined year is string-formatted and not needed for training since we have continuous time_since_business_start var.
del df["joined_year"]

# label encoding to make categorical variables easier to handle:
le = preprocessing.LabelEncoder()
categorical_vars = ["nationality", "passportType", "created_weekday"]
for var in categorical_vars:
    df[var] = le.fit_transform(df[var])


# split data set into training and test set:
train, test = train_test_split(df_nationalised, test_size = 0.2)

# for future use, set up target classification values to score against:
train_target = train["signupCompleted"]
test_target = test["signupCompleted"]
entire_set_target = df["signupCompleted"]

# delete signupCompleted row before classifier is trained
del test["signupCompleted"], train["signupCompleted"]

### First try: Decision tree learner
# Easy to fit - handles categorical vars well, low dims of data, strong exclusion principles (e.g.passport==NULL)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit([df["birthyear"],df["timedelta"],df["passportType"],df["nationality"]],df["signupCompleted"])

### Second: Random Forest
# Next try Random Forest instead of simple decision tree to control overfitting
clf = RandomForestClassifier(100)
clf.fit(train, train_target)



if use_trained_model:
    # use persistent model information from last run to classify new test file
    del df["signupCompleted"]
    pretrained_classifier = pickle.load(open( "rf_model.p", "rb" ))
    accuracy = pretrained_classifier.score(df,entire_set_target)
    print "Pretrained classifier accuracy for new test set: {0}".format(accuracy)

else:
    # as default option, train and test classifier on original test data
    ## Validate:
    print "testing on split within training set"
    print "random forest training set accuracy: {0}".format(clf.score(train, train_target))
    print "random forest test set accuracy: {0}".format(clf.score(test, test_target))

    ##### multiple validation of score to see chance of overfitting

    cv = ShuffleSplit(df["signupCompleted"].count(), n_iter=3,test_size=0.2, random_state=0)

    target = df["signupCompleted"]
    del df["signupCompleted"]
    print "Cross validation mean accuracy: {0}".format(cross_val_score(clf, df, target, cv=cv).mean())

    # 0.962 - mean score on shuffle validation
    ### Store classifier fit to entire test class used:
    full_set_trained_classifier = RandomForestClassifier(100)
    full_set_trained_classifier.fit(df, entire_set_target)
    pickle.dump(full_set_trained_classifier, open( "rf_model.p", "wb" ))
    print "Stored classifier trained on full training dataset as pickle file - 'rf_model.p'"


##############################################



###### III - Miscellaneous code snippets, not needed ####

###### test naive predictor based on single feature (whether *any* Passport registered or not):
#signup_predicted = clf.predict(test)
#signup_predicted_naive = test["passportType"].apply(lambda x: 0 if x == "NULL" else 1)
#signup_predicted_combined = signup_predicted_naive & signup_predicted
#print "dec tree:   ", sklearn.metrics.accuracy_score(test_target, clf.predict(test))
#print "naive:   ", sklearn.metrics.accuracy_score(test_target, signup_predicted_naive)
#print "combined:    ", sklearn.metrics.accuracy_score(test_target, signup_predicted_combined)

###### try SVM -
# perhaps not a natural fit for this kind of low-dim, seasonal, high-categ. data, but worth a try?
#svn_clf = svm.SVC()
#svn_clf.fit(train, train_target)
#print "svn forest training set result", svn_clf.score(train, train_target)
#print "svn forest test set result", svn_clf.score(test, test_target)
# results originally disappointing, got better after deleted misleading vars like last_name

###### Visualise decision tree, rev. transform labels
# import sklearn.metrics
# from sklearn.decomposition import PCA
# from sklearn.externals import joblib

# tree.export_graphviz(clf, out_file='tree.dot')
# print zip(train.columns[clf.tree_.feature], clf.tree_.threshold, clf.tree_.children_left, clf.tree_.children_right)

#pca = PCA(n_components=4)
#pca.fit(train)
#print pca.components_

