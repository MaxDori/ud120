def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from ClassifyNB import classify
    
    ### create classifier and
    ### fit the classifier on the training features and labels
    clf = classify(features_train, labels_train)
    ### use the trained classifier to predict labels for the test features
    
    # place for time
    pred = clf.predict(features_train)
    
    
    
    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = clf.score(features_train, labels_train)
    return accuracy
