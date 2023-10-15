from time import sleep

from src.tradingbot.rabbit_connector import publish_message, create_channel, create_connection


def wait_for_result(
        anchor,
        expected,
        tries=0,
        retry_after=.6
):
    if anchor == expected or tries > 5:
        assert anchor == expected
    else:
        sleep(retry_after)
        return wait_for_result(anchor, expected, tries + 1)


def setup_listener(channel, on_message_callback: callable, queue: str, exchange: str, prefetch_count=1, durable=False):
    channel.queue_declare(queue=queue, durable=durable)
    channel.queue_bind(queue=queue, exchange=exchange)
    channel.basic_qos(prefetch_count=prefetch_count)
    channel.basic_consume(queue=queue, on_message_callback=on_message_callback)

    return channel


def test_rabbitmq_send_message():
    # Should be able to send a message in a given queue
    calls = []
    expected = ['MORTY']
    rabbitmq_channel = create_channel(create_connection())

    def mocked_handler(ch, method, props, body):
        calls.append(body.decode('utf-8'))
        ch.basic_ack(delivery_tag=method.delivery_tag)
        ch.close()

    rabbitmq_channel = setup_listener(channel=rabbitmq_channel, on_message_callback=mocked_handler, queue='hello',
                                      durable=False, exchange='')

    publish_message(channel=rabbitmq_channel, message=b'MORTY', queue='hello')
    rabbitmq_channel.start_consuming()
    wait_for_result(calls, expected)
