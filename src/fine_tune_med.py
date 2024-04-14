from api import client

import os, sys, json
from collections import defaultdict

file_path = os.path.join(
    os.path.dirname(__file__),
    'data/data2.jsonl'
)

with open(file_path, 'r', encoding='utf-8') as f:
    try:
        dataset = [json.loads(line) for line in f]
    except:
        raise ValueError(
            "The dataset must be a valid JSONL file"
        )


size = len(dataset)

if size < 10:
    raise ValueError(
        "The dataset must contain at least 10 examples"
    )

format_errors = defaultdict(int)

for line in dataset:
    if not isinstance(line, dict):
        format_errors["data_type"] += 1
        continue

    messages = line.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue

    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

    valid_keys =(
        "role",
        "content",
        "name",
        "function_call"
    )

    if any(k not in valid_keys for k in message):
        format_errors["message_unrecognized_key"] += 1

    valid_roles = (
        "system",
        "user",
        "assistant",
        "funciton"
    )

    if message.get("role", None) not in valid_roles:
        format_errors["unrecognized_role"] += 1

    conttent = message.get("content", None)

    function_call = message.get("function_call", None)

    if (not conttent and not function_call) or not isinstance(conttent, str):
        format_errors["missing_content"] += 1

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1


if format_errors:
    print("Found errors:")
    for k,v in format_errors.items():
        print(f"{k}: {v}")

    raise ValueError(
        "The dataset containers errors"
    )

uploaded = client.files.create(
    file=open(
        file_path,
        "rb"
    ),
    purpose="fine-tune"
)

file_id = uploaded.id

model = "gpt-3.5-turbo"
fine_tune_job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model=model
)

print("Validating files in progress")
while fine_tune_job.status == "validating_files":
    fine_tune_job = client.fine_tuning.jobs.retrieve(
        fine_tune_job.id
    )

    sys.stdout.write("...")
    sys.stdout.flush()

print("Fine-tuning in progress")

while fine_tune_job.status == "running" or fine_tune_job.status == "queued":
    fine_tune_job = client.fine_tuning.jobs.retrieve(
        fine_tune_job.id
    )
    sys.stdout.write("...")
    sys.stdout.flush()

print("Fine-tuning complete")
print("The name of the new model is: " + fine_tune_job.fine_tuned_model)


