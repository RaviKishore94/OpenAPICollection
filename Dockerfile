FROM python:3.11.0

WORKDIR /user/src/app

COPY './requirements.txt' .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "app.py"]