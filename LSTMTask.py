from numpy.testing import verbose
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
with open('shakespeare.txt', 'r',encoding='utf-8') as file:
    text = file.read().lower()

print("Text length:", len(text))
print("Text sample:", text[:100])

# Text Preprocessing
text = re.sub(r'[^a-z\s]', '', text)
print("Preprossing Text length:",len(text))

# Tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {c:i for i,c in enumerate(chars)}
idx_to_char = {i:c for i,c in enumerate(chars)}

print("Vocabulary size:",vocab_size)


# create input and output sequences
seq_length = 100
x = []
y = []

for i in range(len(text)-seq_length):
    x.append(text[i:i+seq_length])
    y.append(text[i+seq_length])
print("Total Sequences:",len(x))

# encode data
x_encoded = np.array([[char_to_idx[c] for c in seq] for seq in x])
y_encoded = np.array([[char_to_idx[c] for c in seq] for seq in y])

y_encoded = to_categorical(y_encoded,num_classes=vocab_size)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size,output_dim=128,input_length=seq_length),
    LSTM(256,return_sequences=True),
    LSTM(256),
    Dense(vocab_size,activation='softmax')
])
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001)
)

model.summary()

# Train the model
early_stopping = EarlyStopping(monitor="val_loss",patience=3)

model.fit(
    x_encoded,
    y_encoded,
    batch_size=128,
    epochs=5,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Text Generation Function
def generate_text(seed_text,length=300):
    seed_text = seed_text.lower()
    generated_text = seed_text
    for _ in range(length):
        input_seq = seed_text[-seq_length:]
        input_encoded = np.zeros((1, seq_length))
        for t,char in enumerate(input_seq):
            if char in char_to_idx:
                input_encoded[0,t] = char_to_idx[char]
        prediction = model.predict(input_encoded,verbose=0)
        next_char = idx_to_char[np.argmax(prediction)]
        generated_text += next_char
        seed_text += next_char
    return generated_text

# Generatre Text
print(generate_text("To be or not to be ",500))

            
