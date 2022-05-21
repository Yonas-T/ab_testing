FROM python:3.9
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
#RUN python ./train.py
COPY model .
COPY train.py ./train.py
EXPOSE 5000
CMD ["mlflow ui"]