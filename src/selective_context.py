import json
import click

from src.api import get_embedding, client
from src.utils import preprocess_text, cosine_similarity

context_window = 2
model = "gpt-3.5-turbo"
system_prompt = "You are helpful assistant."
history_file_path = "data/context.txt"

full_history = []

global_context = [
    {
        "role": "system",
        "content": system_prompt
    }
]

with open(history_file_path, "w") as file:
    pass

def save_history_to_file(history):
    with open(history_file_path, "w") as f:
        f.write(json.dumps(history))


def load_history_from_file():
    with open(history_file_path, "r") as f:
        try:
            history = json.loads(f.read())
            return history
        except json.JSONDecodeError:
            return []


def sort_history(history, prompt, context_window):
    sorted_history = []
    for segment in history:
        content = segment['content']
        preprocessed_content = preprocess_text(content)
        preprocessed_prompt = preprocess_text(prompt)

        embedding_model = "text-embedding-ada-002"
        embedding_content = get_embedding(
            preprocessed_content,
            embedding_model
        )
        embedding_prompt = get_embedding(
            preprocessed_prompt,
            embedding_model
        )
        similarity = cosine_similarity(
            embedding_content,
            embedding_prompt
        )

        sorted_history.append(
            (segment, similarity)
        )
    sorted_history = sorted(
        sorted_history,
        key=lambda x: x[1],
        reverse=True
    )
    sorted_history = [
        x[0] for x in sorted_history
    ]
    return sorted_history[:context_window]

while True:
    request = input(
        click.style(
            "Input: (type 'exit' to quit):",
            fg="green"
        )
    )

    if request.lower() in ["exit", "quit"]:
        break

    user_prompt = {
        "role": "user",
        "content": request
    }

    full_history = load_history_from_file()
    sorted_history = sort_history(
        full_history,
        request,
        context_window
    )

    sorted_history.append(user_prompt)
    messages = global_context + sorted_history

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=1,
    )

    click.echo(
        click.style(
            "History:",
            fg="blue"
        ) + str(json.dumps(messages, indent=4))
    )

    content = response.choices[0].message.content.strip()

    click.echo(
        click.style(
            "Output: ",
            fg="yellow"
        ) + content
    )

    full_history.append(user_prompt)

    full_history.append({
        "role": "assistant",
        "content": content
    })

    save_history_to_file(full_history)


