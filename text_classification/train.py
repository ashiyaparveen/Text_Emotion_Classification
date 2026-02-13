import pandas as pd

train_data = pd.read_csv('data/train.txt', sep=';', header=None, names=['text', 'label'])
val_data = pd.read_csv('data/val.txt', sep=';', header=None, names=['text', 'label'])
test_data = pd.read_csv('data/test.txt', sep=';', header=None, names=['text', 'label'])

print(train_data.head())

import re

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

train_data['text'] = train_data['text'].apply(clean_text)
val_data['text'] = val_data['text'].apply(clean_text)
test_data['text'] = test_data['text'].apply(clean_text)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['text'])
vocab_size = len(tokenizer.word_index) + 1

train_sequences = tokenizer.texts_to_sequences(train_data['text'])
val_sequences = tokenizer.texts_to_sequences(val_data['text'])
test_sequences = tokenizer.texts_to_sequences(test_data['text'])

max_length = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_length)
val_sequences = pad_sequences(val_sequences, maxlen=max_length)
test_sequences = pad_sequences(test_sequences, maxlen=max_length)

import numpy as np
from keras.utils import to_categorical

train_labels = to_categorical(train_data['label'])
val_labels = to_categorical(val_data['label'])

train_loader = (train_sequences, train_labels)
val_loader = (val_sequences, val_labels)

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_sequences, train_labels, validation_data=(val_sequences, val_labels), epochs=5, batch_size=64)

loss, accuracy = model.evaluate(test_sequences, to_categorical(test_data['label']))
print(f'Test Accuracy: {accuracy * 100:.2f}%')

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

predictions = model.predict(test_sequences)

cm = confusion_matrix(test_data['label'], np.argmax(predictions, axis=1))
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
