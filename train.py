import os
import pickle
from tkinter import filedialog
import tkinter as tk

import numpy as np
from keras.src.optimizers import Adam
from keras.src.saving import get_custom_objects, load_model
from matplotlib import lines
from tqdm import tqdm
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from keras import Model
from keras.src.applications.vgg16 import VGG16, preprocess_input
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import load_img, img_to_array, pad_sequences, to_categorical, plot_model

BASE_DIR = 'flickr8k'
WORKING_DIR = 'working'


# Load VGG16 model
def create_vgg16_model():
    model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


# Extract features from image using VGG16
def extract_features():
    features = {}
    directory = os.path.join(BASE_DIR, 'Images')
    features_file = os.path.join(WORKING_DIR, 'features.pkl')

    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            features = pickle.load(f)
        print("Features loaded from pickle file.")
    else:
        model = create_vgg16_model()
        for img_name in tqdm(os.listdir(directory)):
            img_path = os.path.join(directory, img_name)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            image_id = img_name.split('.')[0]
            features[image_id] = feature

        with open(features_file, 'wb') as f:
            pickle.dump(features, f)
        print("Features extracted and saved to pickle file.")

    return features


# Load captions and clean the text
def load_and_clean_captions():
    with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
        next(f)  # Skip the first line
        captions_doc = f.read()

    mapping = {}
    for line in tqdm(captions_doc.split('\n')):
        tokens = line.split(',')
        if len(tokens) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)

    clean(mapping)
    return mapping


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', ' ')
            import re
            caption = re.sub(r'\s+', ' ', caption)
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption


# Tokenize the captions
def tokenize_captions(mapping):
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in all_captions)

    return tokenizer, vocab_size, max_length


# Data generator for training


def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    # store the sequences
                    X1.append(features[key][0])  # image features
                    X2.append(in_seq)            # text sequence
                    y.append(out_seq)            # next word

            # Once batch is full, yield the data
            if n == batch_size:
                # Return the data as a dictionary for the model
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield {'image': X1, 'text': X2}, y
                X1, X2, y = list(), list(), list()
                n = 0

# Model architecture
def create_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,), name="image")
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name="text")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


# Train the model
def train_model(model, train, mapping, features, tokenizer, max_length, vocab_size, batch_size, epochs=20):
    steps = len(train) // batch_size

    for i in range(epochs):
        # create data generator
        generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
        # fit for one epoch
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# Save model and tokenizer
def save_model_and_tokenizer(model, tokenizer):
    model.save('image_captioning_model.h5')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Model and tokenizer saved.")

import tensorflow as tf
from keras.src.ops import NotEqual
def load_model_and_tokenizer():
    model_file = 'image_captioning_model.h5'
    try:
        with tf.keras.utils.custom_object_scope({'NotEqual': NotEqual}):
            model = load_model(model_file)
            print(f"Mô hình '{model_file}' đã được tải thành công.")
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Tải tokenizer
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        print("Mô hình và tokenizer đã được tải.")
        return model, tokenizer

    except ValueError as e:
        print(f"Lỗi khi tải mô hình: {e}")



# Generate captions for new images
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Generate caption for an image
def generate_caption(image_name, model, tokenizer, features, mapping, max_length):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)


def load_or_train_model(features, mapping, tokenizer, vocab_size, max_length):
    model_file = 'image_captioning_model.h5'

    # Kiểm tra nếu model đã tồn tại
    if os.path.exists(model_file):
        # Nếu model đã tồn tại, tải model và tokenizer
        with tf.keras.utils.custom_object_scope({'NotEqual': NotEqual}):
            model = load_model(model_file)
            print(f"Mô hình '{model_file}' đã được tải thành công.")
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        print("Model và tokenizer đã được tải.")
    else:
        # Nếu model chưa tồn tại, tạo và huấn luyện model
        print("Model chưa có, tiến hành huấn luyện...")
        model = create_model(vocab_size, max_length)

        # Chia dữ liệu thành train và test set
        image_ids = list(mapping.keys())
        split = int(len(image_ids) * 0.90)
        train = image_ids[:split]

        # Huấn luyện model
        train_model(model, train, mapping, features, tokenizer, max_length, vocab_size, batch_size=32, epochs=20)

        # Lưu model và tokenizer
        save_model_and_tokenizer(model, tokenizer)

    return model, tokenizer
# Main workflow
# Hàm mở cửa sổ chọn ảnh

def main():
    features = extract_features()
    mapping = load_and_clean_captions()
    tokenizer, vocab_size, max_length = tokenize_captions(mapping)
    # Tokenize captions
    tokenizer, vocab_size, max_length = tokenize_captions(mapping)

    # Kiểm tra và tải model hoặc huấn luyện model nếu chưa có
    model, tokenizer = load_or_train_model(features, mapping, tokenizer, vocab_size, max_length)


    generate_caption("667626_18933d713e.jpg", model, tokenizer, features, mapping, max_length)


if __name__ == "__main__":
    main()
