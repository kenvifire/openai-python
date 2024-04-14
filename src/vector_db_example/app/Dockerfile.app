FROM python:3.9-bookworm
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD [
    "flask",
     "--app",
     "app",
     "run",
     "--host=0.0.0.0",
     "--port=5000",
     "--debug",
     "--reload",
]