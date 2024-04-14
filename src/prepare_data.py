import json,re
from langdetect import detect
from unidecode import unidecode
import traceback

data = []


def clean_text(text):
    text = unidecode(text)
    text = re.sub(
        r'https?://\S+|www\.\S+', '',
        text
    )

    text = re.sub(
        r'\s*([,.!?]\s*)', r'\1 ',
        text
    )

    text= text.strip()

    text = re.sub(
        r'([:,.!?])([^\s])',
        r'\1 \2',
        text
    )

    return text


open('data/data2.jsonl', 'w').close()

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
the path to well-being"

with open('data/data.json', 'r') as file:
    for line in file:
        json_line = json.loads(line)
        context = json_line['Context']
        response = json_line['Response']

        try:
            if len(context) > 0 and len(response.split()) > 10:
                if detect(context) == 'en' and detect(response) == "en":
                    system = {
                        "role": "system",
                        "content": system_prompt
                    }

                    user = {
                        "role": "user",
                        "content": clean_text(context)
                    }

                    assistant = {
                        "role": "assistant",
                        "content": clean_text(response)
                    }

                    messages = {
                        "messages": [system, user, assistant]
                    }
                    with open('data/data2.jsonl', 'a') as file2:
                        file2.write(json.dumps(messages) + "\n")
        except Exception as ex:
            print(f"Error:\n Context: {context}\n Response: {response}\n, Error{type(ex)}\n")
            traceback.print_exc()

