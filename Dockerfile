FROM  python:3.11-slim
LABEL maintainer="vassilis.liatsos@nufintech.com"

RUN   apt-get update -yqq && apt-get install -yqq --no-install-recommends git

WORKDIR /app
# Set timezone
ENV   TZ=America/New_York
RUN   ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY  requirements.txt ./
RUN   pip install -r requirements.txt

COPY  . /app

RUN mkdir -p /airflow/xcom

ENTRYPOINT ["python3"]