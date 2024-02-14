FROM python:3.9.16-slim

RUN pip install --upgrade pip

COPY ./apisrc /app/src
COPY ./requirements.txt /app

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.service:app", "--host=0.0.0.0", "--reload"]