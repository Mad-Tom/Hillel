from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import Sequential

from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.callbacks import Callback
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils.vis_utils import plot_model

import pandas as pd
import numpy as np

data = pd.read_csv('./data/cleaned_text.csv')
print(data.shape)

data.head()

#взяли текст и привели к виду в пространстве последовательности слов
MAX_NB_WORDS = 80000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(str(data['cleaned_text']))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data['cleaned_text'],
                                                    data['sentiment'],
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=data['sentiment'])

#x_train[15]
#tokenizer.texts_to_sequences([x_train[15]])

train_sequences = tokenizer.texts_to_sequences(x_train.values.astype('U'))
test_sequences = tokenizer.texts_to_sequences(x_test.values.astype('U'))

MAX_LENGTH = 35
padded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH)
padded_test_sequences = pad_sequences(test_sequences, maxlen=MAX_LENGTH)

def get_simple_rnn_model():
    embedding_dim = 300
    embedding_matrix = np.random.random((MAX_NB_WORDS, embedding_dim))

    inp = Input(shape=(MAX_LENGTH,))
    x = Embedding(input_dim=MAX_NB_WORDS, output_dim=embedding_dim, input_length=MAX_LENGTH,
                  weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


rnn_simple_model = get_simple_rnn_model()

filepath="./data/rnn_no_embeddings/weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

batch_size = 256
epochs = 2

history = rnn_simple_model.fit(x=padded_train_sequences,
                    y=y_train,
                    validation_data=(padded_test_sequences, y_test),
                    batch_size=batch_size,
                    callbacks=[checkpoint],
                    epochs=epochs,
                    verbose=1)

best_rnn_simple_model = load_model('data/rnn_no_embeddings/weights-improvement-01-0.5724.hdf5')

y_pred_rnn_simple = best_rnn_simple_model.predict(padded_test_sequences, verbose=1, batch_size=2048)

y_pred_rnn_simple = pd.DataFrame(y_pred_rnn_simple, columns=['prediction'])
y_pred_rnn_simple['prediction'] = y_pred_rnn_simple['prediction'].map(lambda p: 1 if p >= 0.5 else 0)
y_pred_rnn_simple.to_csv('./data/y_pred_rnn_simple.csv', index=False)

y_pred_rnn_simple = pd.read_csv('./data/y_pred_rnn_simple.csv')

from sklearn.metrics import accuracy_score, auc, roc_auc_score
print(accuracy_score(y_test, y_pred_rnn_simple))