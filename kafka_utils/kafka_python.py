import json
import traceback
import glog as logger
import pickle as pkl
import base64
from kafka import KafkaConsumer, KafkaProducer, TopicPartition

def connect_kafka_producer(bootstrap_servers):
    _producer = None
    try:
        _producer = KafkaProducer(bootstrap_servers=bootstrap_servers, api_version=(2,4,0),
                                    # value_serializer=lambda m: pkl.dumps(m))
                                  value_serializer=lambda v: json.dumps(v).encode('utf-8'))
                                # value_serializer=lambda v: base64.b64encode(v))
    except Exception as ex:
        logger.error('Exception: {}'.format(ex))
        logger.error('Traceback: {}'.format(str(traceback.format_exc())))
    finally:
        return _producer


def publish_message(producer_instance, topic_name, partition, message):
    producer_instance.send(topic=topic_name, partition=partition, value=message)
    producer_instance.flush()


def consume_messages(bootstrap_servers, group_id, topic, partition, consumer_timeout_ms = None):
    _consumer = None
    try:
        if consumer_timeout_ms is None:
            _consumer = KafkaConsumer(
                group_id=group_id,
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='earliest',
                enable_auto_commit=False,
                value_deserializer=lambda m: json.loads(m)
            )
        else:
            _consumer = KafkaConsumer(
                group_id=group_id,
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='earliest',
                enable_auto_commit=False,
                value_deserializer=lambda m: json.loads(m),
                consumer_timeout_ms=consumer_timeout_ms
            )
        _consumer.assign([TopicPartition(topic, partition)])
    except Exception as ex:
        logger.error('Exception: {}'.format(ex))
        logger.error('Traceback: {}'.format(str(traceback.format_exc())))
    finally:
        return _consumer

