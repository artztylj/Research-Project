import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Predicting whether to buy [RATIO_PREDICT] in [FUTURE] days 
#based on past [SEQ_LEN] days
SEQ_LEN = 5
FUTURE = 1
RATIO_PREDICT = "AAPL"
EPOCHS = 100
BATCH_SIZE = 25
OPT = tf.keras.optimizers.SGD(learning_rate=0.001, decay=1e-6)
LOSS = "sparse_categorical_crossentropy"

#Random seeds
seeds = [11293, 22159, 15640, 46888, 29298, 8373, 11404, 24723, 43940, 27041, 34130, 37068, 21876, 9958, 11043, 26447,
        34058, 22316, 39085, 41339, 16196, 6329, 9693, 29335, 18796, 15567, 40615, 37105, 2328, 30293, 29674, 19775, 7514,
        32215, 32257, 4523, 30687, 18439, 30170, 39657, 38207, 32804, 20740, 48204, 24434, 26912, 42574, 4669, 35927, 12682,
        13340, 43589, 38361, 28768, 463, 41082, 11634, 32348, 23026, 2137, 4487, 11798, 12764, 2390, 33823, 21184, 46175,
        12447, 35345, 45677, 28306, 21162, 34181, 22054, 11299, 36657, 28433, 18254, 4354, 4770, 44921, 18863, 24900, 28539,
        38615, 37717, 25414, 16738, 8416, 6383, 38653, 17229, 22208, 33494, 19966, 20124, 39472, 42179, 29211, 29670]


#Filenaming conventions for logs directory
time_curr = int(time.time())
time_str = str(time_curr)

NAME = f"{SEQ_LEN}-SEQLEN-{FUTURE}-FUTLEN-{time_str[-5:]}"

#Classification of whether or not the [RATIO_PREDICT] is higher or lower in the future
#current: Current price
#future: Price in [FUTURE] days
def classification(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

#Preprocess the dataframe to drop unneeded data and normalize it
# as well as create the sequences for the network
#df: The dataframe to be preprocessed
#UNSURE OF WHETHER THIS IS THE CORRECT DIRECTION, 
# IF WE NEED THE TARGET TO BE THE PRICE IN [FUTURE] DAYS, THEN WE WILL DROP TARGET 
# FROM THE DF IN THIS FUNCTIONINSTEAD OF FUTURE AND PREDICT FUTURE?
def preprocess_dataframe(df):
    df = df.drop('future', axis=1)

    #If the column is not the target column, we will calculate the percent change for that column
    # and scale the values from 0 to 1 based on the percentage change
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    #Create the sequences of data
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    #Balances the data so that the training and validation sets of data have
    # a roughly equal amount of scenarios where the asset would be bought or sold
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    #Creates the lists that will be returned as well as 
    # fills them with the sequence of features (X) and the labels (Y)
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y



main_df = pd.DataFrame()

#Data preprocessing in order to join all datasets together
#Edit 'ratios' to change which datasets are being used
# ratios: "AAPL", "TSLA" , "BTC-USD", "ETH-USD"
ratios = ['AAPL', 'TSLA']
for ratio in ratios:
    dataset = f"datasets/{ratio}.csv"

    #Renaming columns based on which dataset they came from
    df = pd.read_csv(dataset)
    df.rename(columns={"Open": f"{ratio}_open", "High": f"{ratio}_high", "Low": f"{ratio}_low", "Close": f"{ratio}_close", "Volume": f"{ratio}_volume"}, inplace=True)

    #Indexing based on date
    df.set_index("Date", inplace=True)
    df = df[[f"{ratio}_open", f"{ratio}_high", f"{ratio}_low", f"{ratio}_close", f"{ratio}_volume"]]

    #Joining the dataframes together
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

#Add 'future' column to dataframe that
# signifies the price in [FUTURE] days for [RATIO_PREDICT]
main_df['future'] = main_df[f"{RATIO_PREDICT}_close"].shift(-FUTURE)

#Adds 'target' column to dataframe that signifies whether the price will be higher or lower in [FUTURE] days
main_df['target'] = list(map(classification, main_df[f"{RATIO_PREDICT}_close"], main_df["future"]))

#Sort the data based on date and calculate the day that
# corresponds to the last 5% of data for our entire dataset
dates = sorted(main_df.index.values)
last_5pct = dates[-int(0.05*len(dates))]

#Separates the main dataframe to a validation dataframe that includes
# the last 5% of days and removes those dates from the main dataframe
validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

#Preprocess the data and splits into features and labels 
# for both the training and validation data
x_train, y_train = preprocess_dataframe(main_df)
x_val, y_val = preprocess_dataframe(validation_main_df)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_val = np.asarray(x_val)
y_val = np.asarray(y_val)

#Creating the model
model = tf.keras.Sequential()

glorot_initializer = tf.keras.initializers.glorot_uniform(seed=seeds[0])
orthogonal_initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=seeds[0])

#Two identical layers of a 300 neuron LSTM (GPU) each with a 0.2 dropout
# and batch normalization in order to keep each layers' input normalized
#  throughout training.
model.add(tf.keras.layers.LSTM(256, input_shape=(x_train.shape[1:]), return_sequences=True, kernel_initializer=glorot_initializer, recurrent_initializer=orthogonal_initializer))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=False, kernel_initializer=glorot_initializer, recurrent_initializer=orthogonal_initializer))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

#Two Dense layers, one for consilidating the network structure which has a
# 0.2 dropout, and one with just two neurons for the final output
model.add(tf.keras.layers.Dense(32, activation="relu", kernel_initializer=glorot_initializer))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(2, activation="softmax", kernel_initializer=glorot_initializer))

#Compiles the model according to the topography we have previously set
# with our corresponding loss, optimizer, and metrics
model.compile(loss="sparse_categorical_crossentropy",
    optimizer=OPT,
    metrics=['accuracy'])

#Creates a tensorboard object for the logs directory so that progress after each epoch
# can be visualised and compared to one another after all training has been completed
tb = TensorBoard(log_dir=f'logs/Test/{NAME}')

filepath = "RNN-{epoch:02d}-{val_accuracy:.3f}"
cp = ModelCheckpoint("models/test/{}.hdf5".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))

#Executes the model using our preprocessed data
history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=[tb, cp])