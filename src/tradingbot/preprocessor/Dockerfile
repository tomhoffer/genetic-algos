FROM python:3.11

RUN mkdir /src
WORKDIR /src

RUN apt-get update && apt-get -y install cron

COPY src/tradingbot/preprocessor/requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY src/tradingbot/preprocessor/crontab.txt /etc/cron.d/crontab
RUN chmod 0644 /etc/cron.d/crontab

COPY . .

# Create the log file
RUN touch /var/log/cron.log

# Apply the crontab file
RUN crontab /etc/cron.d/crontab

CMD ["/bin/bash", "-c", "cron && tail -f /var/log/cron.log"]