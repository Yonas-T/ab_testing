FROM python:3.9
WORKDIR /app
RUN pip install -r requiremts.txt
#RUN python ./train.py
COPY . .
EXPOSE 5000
CMD ["mlflow ui"]