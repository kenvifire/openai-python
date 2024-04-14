from api import client
import click

model = "ft:gpt-3.5-turbo-0125:personal::9DjhBYyp"

base_messages = [
    {
        "role": "system",
        "content": "You are a smart home assistant."
    }
]

while True:
    messages = base_messages.copy()

    request = input(
        click.style(
            "Input: (type 'exit' to quit): ",
            fg="green"
        )
    )

    if request.lower() in ["exit", "quit"]:
        break

    messages.append(
        {
            "role": "user",
            "content": f"{request}"
        }
    )

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              max_tokens=200,
                                              temperature=0)

    content = response.choices[0].message.content.strip()

    click.echo(
        click.style("Output: ", fg="yellow") + content
    )

    click.echo()