# напиши скрипт python распознование имён на основе метода байеса

# Разберу задачу пошагово и предоставлю полный скрипт для распознавания имён с использованием наивного байесовского классификатора.
# Шаг 1. Импорт библиотек
# python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import numpy as np

# Шаг 2. Подготовка данных

# Создадим обучающий набор данных — список слов с метками (1 — имя, 0 — не имя):
# python

# Пример обучающих данных
data = [
    ('Анна', 1), ('Мария', 1), ('Иван', 1), ('Пётр', 1), ('Елена', 1),
    ('Сергей', 1), ('Ольга', 1), ('Дмитрий', 1), ('Наталья', 1), ('Алексей', 1),
    ('дом', 0), ('стол', 0), ('книга', 0), ('компьютер', 0), ('машина', 0),
    ('город', 0), ('улица', 0), ('река', 0), ('гора', 0), ('лес', 0),
    ('Александр', 1), ('Виктория', 1), ('Максим', 1), ('София', 1), ('Артём', 1),
    ('школа', 0), ('работа', 0), ('отдых', 0), ('путешествие', 0), ('музыка', 0)
]

# Преобразуем в DataFrame
df = pd.DataFrame(data, columns=['word', 'is_name'])

# Шаг 3. Создание и обучение модели

# Используем пайплайн, объединяющий векторизацию текста и классификатор:
# python

# Создаём пайплайн: векторизация + наивный байес
name_classifier = Pipeline([
    ('vectorizer', CountVectorizer(
        analyzer='char',      # Анализируем по символам
        ngram_range=(2, 3), # Биграммы и триграммы
        lowercase=True        # Приводим к нижнему регистру
    )),
    ('classifier', MultinomialNB())
])

# Разделяем данные на признаки и метки
X = df['word']
y = df['is_name']

# Разбиваем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Обучаем модель
name_classifier.fit(X_train, y_train)

# Шаг 4. Оценка качества модели

# Проверим, насколько хорошо модель научилась распознавать имена:
# python

# Предсказания на тестовой выборке
y_pred = name_classifier.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

# Подробный отчёт
print("\nПодробный отчёт по классификации:")
print(classification_report(y_test, y_pred, target_names=['Не имя', 'Имя']))

# Шаг 5. Функция для распознавания имён в тексте

# Создадим удобную функцию для обработки произвольного текста:
# python

def extract_names(text):
    """
    Извлекает имена из текста с помощью обученной модели.
    
    Args:
        text (str): Текст для анализа
    
    Returns:
        list: Список распознанных имён
    """
    # Очистка текста: оставляем только буквы и пробелы
    cleaned_text = re.sub(r'[^a-zA-Za-яА-Я\s]', '', text)
    
    # Разбиваем на слова и приводим к нижнему регистру
    words = cleaned_text.lower().split()
    
    # Удаляем пустые строки
    words = [word for word in words if word.strip()]
    
    if not words:
        return []
    
    # Предсказание для каждого слова
    predictions = name_classifier.predict(words)
    probabilities = name_classifier.predict_proba(words)
    
    # Собираем только слова, которые модель считает именами (с вероятностью > 0.5)
    names = []
    for word, pred, prob in zip(words, predictions, probabilities):
        if pred == 1 and max(prob) > 0.5:  # Имя с уверенностью > 50%
            # Восстанавливаем оригинальное написание (первая буква заглавная)
            original_word = word.capitalize()
            names.append(original_word)
    
    return names

# Шаг 6. Тестирование модели

# Проверим работу функции на примерах:
# python

# Тестовые тексты
test_texts = [
    "Привет, меня зовут Анна, а это мой друг Иван.",
    "Вчера я был в магазине и купил книгу.",
    "Мария и Сергей пошли в парк с Ольгой.",
    "Компьютер стоит на столе рядом с книгой.",
    "Александр и Виктория отправились в путешествие."
]

print("Результаты распознавания имён:\n")
for text in test_texts:
    names = extract_names(text)
    print(f"Текст: '{text}'")
    print(f"Распознанные имена: {names}")
    print("-" * 50)

# # Полный код скрипта
# # python

# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# import re

# # Подготовка данных
# data = [
#     ('Анна', 1), ('Мария', 1), ('Иван', 1), ('Пётр', 1), ('Елена', 1),
#     ('Сергей', 1), ('Ольга', 1), ('Дмитрий', 1), ('Наталья', 1), ('Алексей', 1),
#     ('дом', 0), ('стол', 0), ('книга', 0), ('компьютер', 0), ('машина', 0),
#     ('город', 0), ('улица', 0), ('река', 0), ('гора', 0), ('лес', 0),
#     ('Александр', 1), ('Виктория', 1), ('Максим', 1), ('София', 1), ('Артём', 1),
#     ('школа', 0), ('работа', 0), ('отдых', 0), ('путешествие', 0), ('музыка', 0)
# ]
# df = pd.DataFrame(data, columns=['word', 'is_name'])

# # Создание и обучение модели
# name_classifier = Pipeline([
#     ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(2, 3), lowercase=True)),
#     ('classifier', MultinomialNB())
# ])
# X = df['word']
# y = df['is_name']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# name_classifier.fit(X_train, y_train)

# # Оценка качества
# y_pred = name_classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Точность модели: {accuracy:.2f}\n")
# print("Подробный отчёт по классификации:")
# print(classification_report(y_test, y_pred, target_names=['Не имя', 'Имя']))

# # Функция для извлечения имён
# def extract_names(text):
#     cleaned_text = re.sub(r'[^a-zA-Za-яА-Я\s]', '', text)
#     words = cleaned_text.lower().split()
#     words = [word for word in words if word.strip()]
#     if not words:
#         return []
#     predictions = name_classifier.predict(words)
#     probabilities = name_classifier.predict_proba(words)
#     names = []
#     for word, pred, prob in zip(words, predictions, probabilities):
#         if pred == 1 and max(prob) > 0.5:
#             original_word = word.capitalize()
#             names.append(original_word)
#     return names

# # Тестирование
# test_texts = [
#     "Привет, меня зовут Анна, а это мой друг Иван.",
#     "Вчера я был в магазине и купил книгу.",
#     "Мария и Сергей пошли в парк с Ольгой."
# ]
# print("\nРезультаты распознавания имён:")
# for text in test_texts:
#     names = extract_names(text)