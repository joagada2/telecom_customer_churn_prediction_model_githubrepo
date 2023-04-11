<<<<<<< HEAD
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app
=======
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn -k uvicorn.workers.UvicornWorker --bind "0.0.0.0:$port" --log-level debug app:app
>>>>>>> b02cc305bbba8ea5b14bdcc346f6c032e385aaea
