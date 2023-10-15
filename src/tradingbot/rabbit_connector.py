import pika
from pika import BlockingConnection
from pika.adapters.blocking_connection import BlockingChannel


def create_connection() -> BlockingConnection:
    credentials = pika.PlainCredentials('user', 'bitnami')
    parameters = pika.ConnectionParameters(credentials=credentials)
    return pika.BlockingConnection(parameters=parameters)


def create_channel(connection: BlockingConnection) -> BlockingChannel:
    return connection.channel()


connection = create_connection()
channel = create_channel(connection)


def publish_message(queue: str, message: bytes, channel: BlockingChannel = channel):
    """
    :param queue: Queue name
    :param message: message
    :param channel: channel to be used
    :return:
    """
    channel.queue_declare(queue=queue, durable=False)
    channel.basic_publish(exchange='',
                          routing_key=queue,
                          body=message)
    connection.close()
