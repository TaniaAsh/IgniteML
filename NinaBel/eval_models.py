#!/usr/bin/env python3
# coding: utf-8
__author__ = "NinaBel"

from sklearn.metrics import accuracy_score, classification_report

def eval_predictions(classifier, X, y):
    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    return {
        'Accuracy': accuracy,
        'Precision (avg)': report['weighted avg']['precision'],
        'Recall (avg)': report['weighted avg']['recall']
    }


def eval_classifier(classifier, X, y, X_test, y_test):
    results = {}
    classifier.fit(X,y)
    results['train'] = eval_predictions(classifier, X, y)
    results['test'] = eval_predictions(classifier, X_test, y_test)
    return results

    
def eval_classifiers(classifiers, target, df, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    
    train_set, test_set = train_test_split(df, 
      test_size=test_size, random_state=random_state, shuffle=False)
    X, y = train_set.drop(columns=[target]), train_set[target]
    X_test, y_test = test_set.drop(columns=[target]), test_set[target]
    results = {}
    for name, classifier in classifiers.items():
        results[name] = eval_classifier(classifier, X, y, X_test, y_test)
    return results