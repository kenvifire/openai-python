from api import client
import click

model = ""
system_prompt = "You are MendMind, an AI Mental Health Coach. \
Your purpose is to support the user through their mental \
health journey with empathy, understanding, and insights \
into managing emotional and psychological challenges. \
While you can provide general advice and emotional \
support, you are not equipped to handle personal contact, \
schedule appointments, or share any specific location \
details. Your only role is to help the user with coping \
strategies, provide information on mental health topics, \
and guide them towards professional resources if needed. \
You can engage in a regular conversation with the user, \
but your primary focus is what you can do best: \
supporting the user with confidentiality and care in \
the path to well-being."

base_messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": "Hi there."
    },
    {
        "role": "assistant",
        "content":
        "My name is MendMind."
        "I'm an AI Mental Health Coach. "
        "How can I help you today?"
    }
]

while True:
    messages = base_messages.copy()
    request = input(
        click.style(
            "Input: (type 'exit' to quit):",
            fg="green"
        )
    )

    if request.lower() in ["exit", "quit"]:
        break

    messages.append({
        "role": "user",
        "content": f"{request}"
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        frequency_penalty=0.5,
        temperature=0.7,
        presence_penalty=0.5,
    )
    content = response.choices[0].message.content.strip()

    click.echo(
        click.style(
            "Output: ",
            fg="yellow"
        ) + content
    )