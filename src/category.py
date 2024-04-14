from src.api import get_embedding
from src.utils import cosine_similarity

categories = [
    'U.S. NEWS',
    'COMEDY',
    'PARENTING',
    'WORD NEWS',
    'CULTURE & ARTS',
    'TECH',
    'SPORTS'
]


def classify_sentence(sentence, model):
    sentence_embedding = get_embedding(
        sentence,
        model
    )
    similarity_scores= {}
    for category in categories:
        category_embeddings = get_embedding(
            category,
            model
        )
        similarity_scores[category] = cosine_similarity(
            sentence_embedding, category_embeddings
        )

    return max(similarity_scores, key=similarity_scores.get)



sentences = [
    "1 dea and 3 injured in E1 Paso, Texas, mall shooting",
    "Director Owen Kline Calls Funny Pasges His 'Self-Critical' Debut",
    "The US is preparing to send more troops to the Middle East"
]

model = "text-embedding-ada-002"

for sentence in sentences:
    category = classify_sentence(sentence, model)

    print(f"'{sentence[:50]}..' => {category}")
    print()




