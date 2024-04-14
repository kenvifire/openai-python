from client import client


def get_embedding(text, model):
    text = text.replace("\n", " ")
    return client.embeddings.create(
        input=[text],
        model=model
    ).data[0].embedding
