import json
import pickle
from collections import defaultdict
import os

class ReverseSearchEngine:
    def __init__(self):
        # Индекс: слово → список (ID документа, позиция в тексте)
        self.index = defaultdict(list)
        # Массив текстов с ID
        self.documents = {}

    def create_index(self, texts):
        """
        Создаёт индексированную базу из массива текстов.

        Args:
            texts (list): список текстовых строк
        """
        self.documents.clear()
        self.index.clear()

        for doc_id, text in enumerate(texts):
            self.documents[doc_id] = text
            words = text.lower().split()

            for position, word in enumerate(words):
                # Удаляем знаки препинания
                clean_word = ''.join(c for c in word if c.isalnum())
                if clean_word:  # Проверяем, что слово не пустое
                    self.index[clean_word].append((doc_id, position))
        print(self.index)
        # exit()

#         1. Создание индексированной базы...
# defaultdict(<class 'list'>, {'привет': [(0, 0)], 'как': [(0, 1), (2, 0)], 'дела': [(0, 2), (2, 2), (4, 0)], 
#                              'у': [(0, 3)], 'меня': [(0, 4)], 'всё': [(0, 5), (2, 4)], 'хорошо': [(0, 6), (2, 5), (4, 1)], 
#                              'сегодня': [(1, 0)], 'хорошая': [(1, 1)], 'погода': [(1, 2), (3, 0), (4, 2)], 
#                              'я': [(1, 3)], 'иду': [(1, 4)], 'гулять': [(1, 5)], 'твои': [(2, 1)], 'надеюсь': [(2, 3)],
#                                'отличная': [(3, 1)], 'можно': [(3, 2)], 'пойти': [(3, 3)], 'в': [(3, 4)],
#                               'парк': [(3, 5)], 'прекрасная': [(4, 3)], 'жизнь': [(4, 4)], 'прекрасна': [(4, 5)]})

        print(f"Создан индекс для {len(texts)} документов")

    def save_index(self, filename):
        """
        Сохраняет индекс и документы в файл.

        Args:
            filename (str): имя файла для сохранения
        """
        data = {
            'index': dict(self.index),
            'documents': self.documents
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Индекс сохранён в файл: {filename}")

    def load_index(self, filename):
        """
        Загружает индекс из файла.

        Args:
            filename (str): имя файла для загрузки
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден")

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.index = defaultdict(list, data['index'])
        self.documents = data['documents']

        print(f"Индекс загружен из файла: {filename}")
        print(f"Загружено документов: {len(self.documents)}")
        print(f"Уникальных слов в индексе: {len(self.index)}")

    def reverse_search(self, word):
        """
        Выполняет обратный поиск слова в проиндексированных текстах.

        Args:
            word (str): искомое слово

        Returns:
            list: список кортежей (ID документа, позиция, текст документа)
        """
        clean_word = word.lower()
        results = []

        if clean_word in self.index:
            for doc_id, position in self.index[clean_word]:
                text = self.documents[doc_id]
                results.append((doc_id, position, text))
        else:
            print(f"Слово '{word}' не найдено в базе")

        return results

    def display_results(self, word, results):
        """
        Отображает результаты поиска в удобном формате.

        Args:
            word (str): искомое слово
            results (list): результаты поиска
        """
        if not results:
            print(f"По запросу '{word}' ничего не найдено")
            return

        print(f"\nРезультаты поиска слова '{word}':")
        print("-" * 50)

        for i, (doc_id, position, text) in enumerate(results, 1):
            print(f"{i}. Документ #{doc_id} (позиция {position}):")
            print(f"   Текст: '{text}'")
            print()


# Пример использования
def main():
    # Исходные тексты
    sample_texts = [
        "Привет, как дела? У меня всё хорошо.",
        "Сегодня хорошая погода, я иду гулять.",
        "Как твои дела? Надеюсь, всё хорошо.",
        "Погода отличная, можно пойти в парк.",
        "Дела хорошо, погода прекрасная, жизнь прекрасна."
    ]

    # Создаём движок поиска
    search_engine = ReverseSearchEngine()

    # 1. Создаём индекс из текстов
    print("1. Создание индексированной базы...")
    search_engine.create_index(sample_texts)

    # 2. Сохраняем индекс в файл
    print("\n2. Сохранение индекса...")
    index_filename = "search_index.pkl"
    search_engine.save_index(index_filename)

    # 3. Загружаем индекс (имитация нового запуска программы)
    print("\n3. Загрузка индекса...")
    new_engine = ReverseSearchEngine()
    new_engine.load_index(index_filename)
    # index_words = new_engine.load_index(index_filename)
    # print(index_words)
    # exit()

    # 4. Выполняем поиск
    print("\n4. Выполнение поиска...")
    search_words = ["хорошо", "погода", "дела", "парк"]
    search_words = ["хорошая", "погода"]

    for word in search_words:
        results = new_engine.reverse_search(word)
        new_engine.display_results(word, results)

if __name__ == "__main__":
    main()