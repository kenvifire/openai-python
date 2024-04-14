from api import client
import os,sys

file_path = os.path.join(
    os.path.dirname(__file__),
    'data/data.jsonl'
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

print("Fine-tuning progress")
while fine_tune_job.status == "running" or fine_tune_job.status == 'queued':
    fine_tune_job = client.fine_tuning.jobs.retrieve(
        fine_tune_job.id
    )
    sys.stdout.write("...")
    sys.stdout.flush()


print("Fine-tuning complete")
print("The name of the new model is:" + fine_tune_job.fine_tuned_model)

