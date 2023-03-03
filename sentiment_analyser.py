import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Load the Twitter Sentiment Analysis dataset
data = pd.read_csv('training.1600000.processed.noemoticon.csv')

# Removing the unnecesarry columns
data.drop(data.columns.difference(['texts']), axis=1, inplace=True)

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['text'])

# Convert text data to sequences
train_sequences = tokenizer.texts_to_sequences(train_data['text'])
test_sequences = tokenizer.texts_to_sequences(test_data['text'])

# Pad the sequences to a fixed length
max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Define the ANN model
model = Sequential()
model.add(Embedding(5000, 32, input_length=max_length))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_data['sentiment'], batch_size=64, epochs=10, validation_data=(test_padded, test_data['sentiment']))

# Predict sentiment for new text
new_text = ['I love this product!', 'I hate this service!', 'This movie was okay.']
new_text_sequences = tokenizer.texts_to_sequences(new_text)
new_text_padded = pad_sequences(new_text_sequences, maxlen=max_length, padding='post', truncating='post')
predictions = model.predict(new_text_padded)

# Print the predicted sentiment for each new text
for i in range(len(new_text)):
    if predictions[i] > 0.5:
        print(new_text[i] + " --> Positive")
    else:
        print(new_text[i] + " --> Negative")
