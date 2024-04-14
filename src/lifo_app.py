from api import client
import click, json

n = 2
model = "gpt-3.5-turbo"

system_prompt = "You are helpful assistant."

global_context = [
    {
        "role": "system",
        "content": system_prompt
    }
]

history_file_path = "data/context.txt"

with open(history_file_path, "w") as file:
    pass

def save_history_to_file(history):

    with open(history_file_path, "w") as f:
        f.write(json.dumps(history))


def load_history_from_file():
    with open(history_file_path, "r") as f:
        try:
            history = json.loads(f.read())
            return history[-n:]

        except json.JSONDecodeError:
            return []


full_history = []

while True:
    request = input(
        click.style(
            "Input: (type 'exit' to quit): ",
            fg="green"
        )
    )

    if request.lower() in ["exit", "quit"]:
        break

    history = {
        "role": "user",
        "content": request
    }

    full_history = load_history_from_file()
    full_history.append(history)
    messages = global_context + full_history

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=150,
        temperature=0.7,
    )

    click.echo(
        click.style("History: ", fg="blue") + str(json.dumps(messages, indent=4))
    )

    content = response.choices[0].message.content.strip()

    click.echo(
        click.style(
            "Output: ",
            fg="yellow"
        ) + content
    )
    full_history.append({
        "role": "assistant",
        "content": content
    })

    save_history_to_file(full_history)

