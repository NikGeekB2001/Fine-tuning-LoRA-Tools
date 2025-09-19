# Файл: analysis_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

class ResultAnalyzer:
    def __init__(self, tester):
        self.tester = tester
    
    def plot_confusion_matrix(self, true_labels, predictions, title="Confusion Matrix"):
        """Строим матрицу ошибок"""
        # Преобразуем в плоские списки
        flat_true = [label for sequence in true_labels for label in sequence]
        flat_pred = [label for sequence in predictions for label in sequence]
        
        # Получаем уникальные метки
        labels = sorted(set(flat_true + flat_pred))
        
        # Строим матрицу
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(flat_true, flat_pred, labels=labels)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('Истинные метки')
        plt.xlabel('Предсказанные метки')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_errors(self, true_labels, predictions):
        """Анализ ошибок по типам"""
        errors = []

        for true_seq, pred_seq in zip(true_labels, predictions):
            for true_label, pred_label in zip(true_seq, pred_seq):
                if true_label != pred_label:
                    errors.append((true_label, pred_label))

        error_counts = Counter(errors)

        print("🔍 Топ-10 самых частых ошибок:")
        for (true_label, pred_label), count in error_counts.most_common(10):
            print(f"  {true_label} -> {pred_label}: {count} раз")

        return error_counts

    def plot_label_distribution(self, labels, title="Распределение меток"):
        """График распределения меток"""
        # Преобразуем в плоский список
        flat_labels = [label for sequence in labels for label in sequence]

        # Считаем частоты
        label_counts = Counter(flat_labels)

        # Строим график
        plt.figure(figsize=(12, 6))
        labels_sorted, counts = zip(*label_counts.most_common())
        plt.bar(labels_sorted, counts)
        plt.title(title)
        plt.xlabel('Метки')
        plt.ylabel('Количество')
        plt.xticks(rotation=45)

        # Добавляем значения на столбцы
        for label, count in zip(labels_sorted, counts):
            plt.text(label, count + max(counts) * 0.01, str(count),
                    ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Демонстрация использования анализатора результатов"""
    print("🔍 Анализатор результатов тестирования LoRA модели")
    print("=" * 60)

    # Импортируем ModelTester из best.py
    from best import ModelTester

    # Создаём тестер
    tester = ModelTester()

    # Загружаем модели
    base_model = tester.load_base_model()
    lora_model = tester.load_lora_model()

    if lora_model is None:
        print("❌ Не удалось загрузить LoRA модель")
        return

    # Получаем тестовые данные
    test_examples = tester.create_test_examples()

    # Оцениваем модели
    print("\n📊 Оценка моделей на тестовых примерах...")

    # Собираем предсказания для анализа
    all_true_labels = []
    all_base_predictions = []
    all_lora_predictions = []

    for example in test_examples:
        text = example["text"]
        expected = example["expected"]

        # Предсказания базовой модели
        base_pred = tester.predict_single_text(base_model, text)
        base_labels = [label for _, label in base_pred]

        # Предсказания LoRA модели
        lora_pred = tester.predict_single_text(lora_model, text)
        lora_labels = [label for _, label in lora_pred]

        # Истинные метки
        true_labels = [label for _, label in expected]

        # Выравниваем длины
        min_len = min(len(base_labels), len(lora_labels), len(true_labels))
        if min_len > 0:
            all_true_labels.append(true_labels[:min_len])
            all_base_predictions.append(base_labels[:min_len])
            all_lora_predictions.append(lora_labels[:min_len])

    if all_true_labels and all_lora_predictions:
        # Создаём анализатор
        analyzer = ResultAnalyzer(tester)

        # Анализ ошибок LoRA модели
        print("\n🔍 Анализ ошибок LoRA модели:")
        analyzer.analyze_errors(all_true_labels, all_lora_predictions)

        # График распределения истинных меток
        print("\n📈 Построение графиков...")
        analyzer.plot_label_distribution(all_true_labels, "Распределение истинных меток")

        print("\n✅ Анализ завершён! Графики сохранены в файлы PNG.")
    else:
        print("❌ Недостаточно данных для анализа")

if __name__ == "__main__":
    main()
