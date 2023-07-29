'''                                                                 File for holding model functions                                                           '''


def get_acc_table(train_X, train_y, validate_X, validate_y, clf_lst):

    ''' take in train data split into X and y, Validate data split into X and y, and list of classifiers
        print table of accuracy scores for train and validate data when run on each classifier in the list'''

    print('Accuracy Scores')
    print('---------------')

    for clf in clf_lst:

        label = str(clf)[:-2]
        
        obj = clf.fit(train_X, train_y)
        train_score = obj.score(train_X, train_y)
        validate_score = obj.score(validate_X, validate_y)

        print(f'{label} Train: {train_score} Validate: {validate_score}')