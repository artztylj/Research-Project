import pandas as pd
from tensorboard.data import experimental

#Gets the dataframe will all information from tensorboard.dev
def extract(experiment_id):
    experiment = experimental.ExperimentFromDev(experiment_id)
    extractedDF = experiment.get_scalars()
    return extractedDF

#Removes the training data from the dataframe so that we only have the validation data
def remove_training(validationDF):
    for i in range(len(validationDF)-1, -1, -1):
        run = validationDF.iloc[i].get('run')
        if 'train' in run:
            validationDF = validationDF.drop(i)
    validationDF = validationDF.reset_index(drop=True)
    return validationDF

#Removes the evaluation data so that we only have accuracy and loss
def remove_evaluation(evaluationDF):
    for i in range(len(evaluationDF)-1, -1, -1):
        tag = evaluationDF.iloc[i].get('tag')
        if tag == "evaluation_accuracy_vs_iterations" or tag == "evaluation_loss_vs_iterations":
            evaluationDF = evaluationDF.drop(i)
    evaluationDF = evaluationDF.reset_index(drop=True)
    return evaluationDF

#Pairs the two above remove functions to one function call
def remove(removedDF):
    removedDF = remove_training(removedDF)
    removedDF = remove_evaluation(removedDF)
    return removedDF

def separate(dfAcc):
    dfLoss = pd.DataFrame()

    for i in range(len(dfAcc)):
        tag = dfAcc.iloc[i].get('tag')
        if 'loss' in tag:
            dfLoss = dfLoss.append(dfAcc.iloc[i])

    for i in range(len(dfAcc)-1, -1, -1):
        tag = dfAcc.iloc[i].get('tag')
        if 'loss' in tag:
            dfAcc = dfAcc.drop(i)

    return [dfAcc, dfLoss]

def unstack(unstackDF):
    finaldf = pd.DataFrame()

    for i in range(88):
        newdf = unstackDF.iloc[:100]
        for i in range(100):
            unstackDF = unstackDF.drop([i])
        unstackDF = unstackDF.reset_index(drop=True)
        newdf = newdf.drop(['tag', 'step'], axis=1)

        finaldf[newdf.iloc[0].get('run')] = newdf.iloc[:100, 1]

    return finaldf



#Experiment id's: 	  Stock Data			   Crypto Data				 Both (Stock)			   Both (Crypto)
experiment_id = ['YeGLuQyzS3qlH5m02SmnFg', 'V1WFVXUySWSPjGOLg3wqoA', '0Yvz3iqYTW2JN1hY7d2CpQ', 'XJxeEOq4Qz6wZTjXojXgag']

