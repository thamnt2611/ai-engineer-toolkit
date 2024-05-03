from confluent_kafka import Producer, Consumer, TopicPartition
from confluent_kafka.schema_registry.json_schema import JSONSerializer
import kafka_python
import numpy as np
import cv2
import base64
from PIL import Image
import io
import json
def connect_kafka_producer(bootstrap_servers):
    _producer = None
    try:
        _producer = Producer({'bootstrap.servers': bootstrap_servers})
        # _producer = KafkaProducer(bootstrap_servers=bootstrap_servers, api_version=(2,4,0),
        #                             # value_serializer=lambda m: pkl.dumps(m))
        #                           value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        #                         # value_serializer=lambda v: base64.b64encode(v))
    except Exception as ex:
        logger.error('Exception: {}'.format(ex))
        logger.error('Traceback: {}'.format(str(traceback.format_exc())))
    finally:
        return _producer

def request_result_callback(err, msg):
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))


def publish_message(producer_instance, topic_name, partition, message):
    producer_instance.poll(0)
    serialized_message = json.dumps(message).encode('utf-8')
    producer_instance.produce(
        topic=topic_name, 
        partition=partition, 
        value=serialized_message, 
        callback=request_result_callback
    )
    producer_instance.flush()
    
    
def consume_messages(bootstrap_servers, group_id, topic, partition, consumer_timeout_ms = None):
    # Not deserialize object
    _consumer = None
    try:
        if consumer_timeout_ms is None:
            _consumer = Consumer(
                {
                    'bootstrap.servers': bootstrap_serverss,
                    'group.id': group_id,
                    'auto.offset.reset': 'earliest',
                    'enable.auto.commit': False,
                }
            )
        else:
            _consumer = Consumer(
                {
                    'bootstrap.servers': bootstrap_serverss,
                    'group.id': group_id,
                    'auto.offset.reset': 'earliest',
                    'enable.auto.commit': False,
                    'consumer.timeout.ms': consumer_timeout_ms
                }
            )
        _consumer.subcribe([TopicPartition(topic, partition)])
    except Exception as ex:
        logger.error('Exception: {}'.format(ex))
        logger.error('Traceback: {}'.format(str(traceback.format_exc())))
    finally:
        return _consumer


def encode_and_transmit_numpy_array_in_bytes(numpy_array:np.array) -> str:
    # Create a Byte Stream Pointer
    compressed_file = io.BytesIO()
    # Use PIL JPEG reduction to save the image to bytes
    Image.fromarray(numpy_array).save(compressed_file, format="JPEG")
    # Set index to start position
    compressed_file.seek(0)
    # Convert the byte representation to base 64 representation for REST Post
    return json.dumps(base64.b64encode(compressed_file.read()).decode())

from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

@timeit
def time_val_confluent_kafka(encoded_image, bootstrap_servers, group_id, topic, partition):
    producer = kafka_python.connect_kafka_producer(bootstrap_servers)
    print(producer)
    message = {
        "image": encoded_image
    }
    for i in range(20):
        kafka_python.publish_message(producer, topic, partition, message)
    
@timeit
def time_val_python_kafka(encoded_image, bootstrap_servers, group_id, topic, partition):
    producer = connect_kafka_producer(bootstrap_servers)
    print(producer)
    message = {
        "image": encoded_image
    }
    for i in range(20):
        publish_message(producer, topic, partition, message) 
        print("publish OK")

if __name__ == "__main__":
    image = cv2.imread("/home/asi/camera/thamnt/dataset/car_color/car_color_custom/train/Black/1_0__1563420627.4476128_15784_2.jpg")
    image = encode_and_transmit_numpy_array_in_bytes(image)
    
    BOOTSTRAP_SERVERS = "10.0.5.14:29092"
    GROUP_NAME = "people_tracking_0"
    TOPIC_NAME = "people_tracking"
    TOPIC_PARTITION = 0
    
    time_val_confluent_kafka(image, 
                            BOOTSTRAP_SERVERS, 
                            GROUP_NAME, 
                            TOPIC_NAME, 
                            TOPIC_PARTITION)
