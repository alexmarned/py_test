# напиши скрипт python распознование имён на основе keras tensorflow

# Разберу создание скрипта для распознавания имён с использованием Keras и TensorFlow пошагово.
# Шаг 1. Установка зависимостей

# Убедитесь, что у вас установлены необходимые библиотеки:
# bash

# pip install tensorflow numpy pandas scikit-learn

# Шаг 2. Импорт библиотек
# python

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Шаг 3. Подготовка данных

# Создадим пример набора данных и подготовим его для обучения:
# python

# Пример данных: список имён и не-имён
data = [
    ('Анна', 'name'),
    ('Иван', 'name'),
    ('Мария', 'name'),
    ('Пётр', 'name'),
    ('Елена', 'name'),
    ('дом', 'not_name'),
    ('стол', 'not_name'),
    ('книга', 'not_name'),
    ('компьютер', 'not_name'),
    ('машина', 'not_name')
]

# Разделение на тексты и метки
texts, labels = zip(*data)

# Кодирование меток
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Токенизация текстов
tokenizer = Tokenizer(char_level=True)  # Работаем на уровне символов
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Пэддинг последовательностей
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding='post')
y = np.array(encoded_labels)

# Шаг 4. Создание и обучение модели
# python

# Параметры модели
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50

# Создание модели
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64, dropout=0.5, recurrent_dropout=0.5),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Бинарная классификация
])

# Компиляция модели
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Вывод структуры модели
model.summary()

# Обучение модели
history = model.fit(
    X, y,
    epochs= 450,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Шаг 5. Сохранение модели и токенизатора
# python

# Сохранение модели
model.save('name_recognition_model.h5')

# Сохранение токенизатора и энкодера меток
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Шаг 6. Функция для предсказания
# python

def predict_name(text, model, tokenizer, label_encoder, max_length):
    """
    Функция для предсказания, является ли текст именем.
    """
    # Преобразование текста в последовательность
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Предсказание
    prediction = model.predict(padded, verbose=0)
    predicted_class = (prediction > 0.5).astype(int)
    
    # Декодирование класса
    label = label_encoder.inverse_transform(predicted_class[0])
    confidence = prediction[0][0] if predicted_class[0][0] == 1 else 1 - prediction[0][0]
    
    return label[0], confidence

# Загрузка модели и вспомогательных файлов для предсказаний
from tensorflow.keras.models import load_model

loaded_model = load_model('name_recognition_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as handle:
    loaded_label_encoder = pickle.load(handle)

# Тестирование
test_names = ['Анна', 'дом', 'Иван', 'стол']
for name in test_names:
    result, confidence = predict_name(
        name, loaded_model, loaded_tokenizer, loaded_label_encoder, max_length
    )
    print(f"Текст: '{name}' -> Класс: {result}, Уверенность: {confidence:.2f}")

# Важные замечания и рекомендации

#     Объём данных. Для реального использования нужен большой набор данных с именами и не-именами.

#     Качество данных. Убедитесь, что данные сбалансированы (примерно равное количество имён и не‑имён).

#     Архитектура. Для сложных задач можно попробовать:

#         двунаправленные LSTM (Bidirectional(LSTM(...)));

#         CNN + LSTM;

#         трансформеры.

#     Предобработка. Можно добавить нормализацию текста (нижний регистр, удаление лишних символов).

#     Оценка. Добавьте метрики качества (precision, recall, F1) для оценки производительности.

#     Кросс‑валидация. Используйте её для более надёжной оценки модели.

# Хотите, я уточню какой‑то из этапов или помогу адаптировать скрипт под ваши конкретные данные?