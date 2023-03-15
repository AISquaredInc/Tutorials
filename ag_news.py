from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import pandas as pd
import numpy as np
import aisquared

train_df = pd.read_csv('ag_news_train.csv')
test_df = pd.read_csv('ag_news_test.csv')

tokenizer = tf.keras.preprocessing.text.Tokenizer(10000, oov_token = 1)
tokenizer.fit_on_texts(train_df.Description)
vocab = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_df.Description)
test_sequences = tokenizer.texts_to_sequences(test_df.Description)
train_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, 64, padding = 'pre', truncating = 'post')
test_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, 64, padding = 'pre', truncating = 'post')

train_indices = np.arange(train_sequences.shape[0])
np.random.shuffle(train_indices)

del vocab[1]

train_labels = train_df['Class Index'].values.astype(int) - 1
test_labels = test_df['Class Index'].values.astype(int) - 1

train_sequences = train_sequences[train_indices]
train_labels = train_labels[train_indices]

input_layer = tf.keras.layers.Input(64)
x = tf.keras.layers.Embedding(
    10000,
    16
)(input_layer)
x = tf.keras.layers.Flatten()(x)
for _ in range(10):
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
output_layer = tf.keras.layers.Dense(4, activation = 'softmax')(x)
model = tf.keras.models.Model(input_layer, output_layer)
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer = tf.keras.optimizers.Adam())
model.summary()

callback = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    min_delta = 0.01,
    restore_best_weights = True
)

model.fit(
    train_sequences,
    train_labels,
    batch_size = 512,
    epochs = 100,
    validation_split = 0.2,
    callbacks = [callback]
)

preds = model.predict(test_sequences).argmax(axis = 1)
print(confusion_matrix(test_labels, preds))
print(classification_report(test_labels, preds))

model.save('topic_classifier.h5')

harvester = aisquared.config.harvesting.InputHarvester()
preprocesser = aisquared.config.preprocessing.text.TextPreprocessor(
    [
        aisquared.config.preprocessing.text.RemoveCharacters(),
        aisquared.config.preprocessing.text.ConvertToCase(lowercase = True),
        aisquared.config.preprocessing.text.Tokenize(),
        aisquared.config.preprocessing.text.ConvertToVocabulary(
            vocab,
            oov_character = 1,
            start_character = 0
        ),
        aisquared.config.preprocessing.text.PadSequences(
            length = 64,
            pad_location = 'pre',
            truncate_location = 'post'
        )
    ]
)
model = aisquared.config.analytic.LocalModel('topic_classifier.h5', 'text')
postprocesser = aisquared.config.postprocessing.MulticlassClassification(
    [
        'World',
        'Sports',
        'Business',
        'Science/Technology'
    ]
)
renderer = aisquared.config.rendering.DocumentRendering(include_probability = True)
model_feedback = aisquared.config.feedback.ModelFeedback()
model_feedback.add_question('Is this model useful?', choices = ['yes', 'no'])
model_feedback.add_question('Please elaborate', answer_type = 'text')
prediction_feedback = aisquared.config.feedback.MulticlassFeedback(
    [
        'World',
        'Sports',
        'Business',
        'Science/Technology'
    ]
)

config = aisquared.config.ModelConfiguration(
    'NewsTopicClassifier',
    harvester,
    preprocesser,
    model,
    postprocesser,
    renderer,
    [model_feedback, prediction_feedback]
).compile(dtype = 'float16')



