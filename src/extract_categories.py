import json
import pandas as pd

from src.category import classify_sentence
from sklearn.metrics import precision_score

categories = set()
with open('data/news.json') as file:
    for line in file:
        data = json.loads(line)
        categories.add(data['category'])

categories = list(categories)


def evaluate_precision(categories):
    df = pd.read_json(
        'data/news.json',
        lines=True,
    ).head(20)
    y_true = []
    yr_pred = []

    model = "text-embedding-ada-002"

    for _, row in df.iterrows():
        real_category = row['category']
        predicted_category = classify_sentence(
            row['headline'],
            model=model
        )

        y_true.append(real_category)
        yr_pred.append(predicted_category)

        if real_category != predicted_category:
            print(
                "[] Incorrect prediction:"
                f"{row['headline'][:50]}...\n"
                f"Real: {real_category[:20]}"
                f"Predicted: {predicted_category}[:20]"
            )
        else:
            print(
                "[] Correct prediction:"
                f"{row['headline'][:50]}...\n"
                f"Real: {real_category[:20]}"
                f"Predicted: {predicted_category[:20]}"
            )

        return precision_score(
            y_true,
            yr_pred,
            average="micro",
            labels=categories
        )


precision = evaluate_precision(categories)
print(f"Precision: {precision}")
