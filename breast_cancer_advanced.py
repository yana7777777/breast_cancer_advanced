import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.datasets import load_breast_cancer

print("Библиотеки загружены")


data = load_breast_cancer(as_frame=True)

# создайте DataFrame
df = data.frame.copy()

print("Размер таблицы:", df.shape)
print("Классы:", list(data.target_names))
display(df.head())

assert df.shape[0] == 569
assert df.shape[1] == 31


# создайте diagnosis через map:
# 0 -> malignant
# 1 -> benign
df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})

class_counts = df["diagnosis"].value_counts()
display(class_counts)

assert set(df["diagnosis"].unique()) == {"malignant", "benign"}
assert class_counts.sum() == len(df)

class_counts.plot(kind="bar", figsize=(6, 4))
plt.title("Количество объектов по классам")
plt.xlabel("Класс")
plt.ylabel("Количество")
plt.grid(True)
plt.tight_layout()
plt.show()


feature = "worst area"

# посчитайте средний mean radius по diagnosis
mean_by_class = df.groupby("diagnosis")[feature].mean().round(2)

display(mean_by_class)

assert "benign" in mean_by_class.index
assert "malignant" in mean_by_class.index
assert mean_by_class["malignant"] > mean_by_class["benign"]


plt.figure(figsize=(8, 4))
for diagnosis in ["benign", "malignant"]:
    values = df[df["diagnosis"] == diagnosis][feature]
    plt.hist(values, bins=20, alpha=0.6, label=diagnosis)

plt.title("Распределение признака worst area")
plt.xlabel(feature)
plt.ylabel("Количество")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# порог = середина между средними benign и malignant
threshold = (mean_by_class["benign"] + mean_by_class["malignant"]) / 2

# prediction по правилу if
df["prediction"] = df[feature].apply(
    lambda x: "malignant" if x > threshold else "benign"
)

print("Порог:", round(threshold, 2))
display(df[[feature, "diagnosis", "prediction"]].head(10))

assert threshold > 0
assert set(df["prediction"].unique()).issubset({"malignant", "benign"})



# посчитайте количество верных ответов
correct = (df["diagnosis"] == df["prediction"]).sum()
total = len(df)
accuracy = correct / total

print("Верных ответов:", correct)
print("Всего объектов:", total)
print("Accuracy:", round(accuracy, 3))

assert correct <= total
assert 0 <= accuracy <= 1
assert accuracy > 0.75


# выберите строки, где diagnosis не равно prediction
mistakes = df[df["diagnosis"] != df["prediction"]]

print("Количество ошибок:", len(mistakes))
display(mistakes[[feature, "diagnosis", "prediction"]].head(10))

assert len(mistakes) > 0
assert len(mistakes) < len(df)

report = pd.DataFrame([
    {"metric": "rows", "value": len(df)},
    {"metric": "feature", "value": feature},
    {"metric": "threshold", "value": round(threshold, 3)},
    {"metric": "accuracy", "value": round(accuracy, 3)},
    {"metric": "mistakes", "value": len(mistakes)},
])

report_path = "block03_simple_breast_cancer_report.csv"

# сохраните отчёт в CSV
...

display(report)
print("Файл сохранён:", report_path)

print("\nВывод:")
print("Мы сделали простой ИИ-подобный алгоритм на одном признаке.")
print("Он работает лучше случайного выбора, но ошибается и не является медицинской диагностикой.")

assert report.shape[0] == 5