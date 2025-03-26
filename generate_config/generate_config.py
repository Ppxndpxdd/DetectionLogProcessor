import configparser
import os


def create_mqtt_config(
    client_id: str,
    username:str,
    password:str,
    broker:str,
    port:int,
    topic:str,
    file_path:str,
    ca_certs_path:str
):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    config = configparser.ConfigParser()
    config['mqtt_credentials'] = {
        'broker': broker,
        'port': port,
        'topic': topic,
        'client_id': client_id,
        'username': username,
        'password': password,
        'ca_certs_path': ca_certs_path
    }
    # Write the configuration to a file
    with open(file_path, 'w') as configfile:
        config.write(configfile)
    
def read_mqtt_config(file_path:str = 'config.ini'):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(file_path)

    # Access values from the configuration file
    broker = config['mqtt_credentials']['broker']
    port = int(config['mqtt_credentials']['port'])
    topic = config['mqtt_credentials']['topic']
    client_id = config['mqtt_credentials']['client_id']
    username = config['mqtt_credentials']['username']
    password = config['mqtt_credentials']['password']
    ca_certs_path = config['mqtt_credentials']['ca_certs_path']

    # Return a dictionary with the retrieved values
    config_values = {
        'broker': broker,
        'port': port,
        'topic': topic,
        'client_id': client_id,
        'username': username,
        'password': password,
        'ca_certs_path': ca_certs_path
    }

    return config_values

        
def create_postgres_config(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    file_path: str = 'config.ini'
):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    config = configparser.ConfigParser()
    config['postgres_credentials'] = {
        'host': host,
        'port': port,
        'database': database,
        'user': user,
        'password': password
    }
    # Write the configuration to a file
    with open(file_path, 'w') as configfile:
        config.write(configfile)

def read_postgres_config(file_path:str = 'config.ini'):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(file_path)

    # Access values from the configuration file
    host = config['postgres_credentials']['host']
    port = int(config['postgres_credentials']['port'])
    database = config['postgres_credentials']['database']
    user = config['postgres_credentials']['user']
    password = config['postgres_credentials']['password']

    # Return a dictionary with the retrieved values
    config_values = {
        'host': host,
        'port': port,
        'database': database,
        'user': user,
        'password': password
    }

    return config_values

def create_weight_config(
    plate_weight_path: str,
    ocr_weight_path: str,
    file_path: str = 'config.ini'
):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    config = configparser.ConfigParser()
    config['model_weight'] = {
        'plate_weight_path': plate_weight_path,
        'ocr_weight_path': ocr_weight_path,
    }
    # Write the configuration to a file
    with open(file_path, 'w') as configfile:
        config.write(configfile)
        
def read_model_weight_config(file_path:str = 'config.ini'):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(file_path)

    # Access values from the configuration file
    default_rtsp_url = config['model_weight']['default_rtsp_url']
    output_dir = config['model_weight']['output_dir']
    buffer_size = int(config['model_weight']['buffer_size'])
    plate_weight_path = config['model_weight']['plate_weight_path']
    ocr_weight_path = config['model_weight']['ocr_weight_path']
    
    # Return a dictionary with the retrieved values
    config_values = {
        'default_rtsp_url': default_rtsp_url,
        'output_dir': output_dir,
        'buffer_size': buffer_size,
        'plate_weight_path': plate_weight_path,
        'ocr_weight_path': ocr_weight_path,
    }

    return config_values

def create_minio_config(
    endpoint: str,
    access_key: str,
    secret_key: str,
    file_path: str = 'config.ini'
):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    config = configparser.ConfigParser()
    config['minio_credentials'] = {
        'endpoint': endpoint,
        'access_key': access_key,
        'secret_key': secret_key
    }
    # Write the configuration to a file
    with open(file_path, 'w') as configfile:
        config.write(configfile)

def read_minio_config(file_path:str = 'config.ini'):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(file_path)

    # Access values from the configuration file
    endpoint = config['minio_credentials']['endpoint']
    access_key = config['minio_credentials']['access_key']
    secret_key = config['minio_credentials']['secret_key']

    # Return a dictionary with the retrieved values
    config_values = {
        'endpoint': endpoint,
        'access_key': access_key,
        'secret_key': secret_key
    }

    return config_values




if __name__ == "__main__":
    input("This script will generate a configuration file for MQTT. Press Enter to continue...")
    broker = input("Enter the MQTT broker address: ")
    port = int(input("Enter the MQTT port: "))
    topic = input("Enter the MQTT topic: ")
    client_id = input("Enter the client ID: ")
    username = input("Enter the username: ")
    password = input("Enter the password: ")
    ca_certs_path = input("Enter the path to the CA certificates file: ")
    file_path = input("Enter the path to save the configuration file: ")
    create_mqtt_config(
        client_id=client_id, 
        username=username, 
        password=password,
        broker=broker,
        port=port,
        topic=topic,
        file_path=file_path,
        ca_certs_path=ca_certs_path
    )
    print("Configuration file generated successfully!")
    
    input("This script will generate a configuration file for Postgres. Press Enter to continue...")
    host = input("Enter the host address: ")
    port = int(input("Enter the port: "))
    database = input("Enter the database: ")
    user = input("Enter the username: ")
    password = input("Enter the password: ")
    file_path = input("Enter the path to save the configuration file: ")
    create_postgres_config(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        file_path=file_path
    )
    print("Configuration file generated successfully!")
    
    input("This script will generate a configuration file for Model Weight. Press Enter to continue...")
    plate_weight_path = input("Enter the local weight path: ")
    ocr_weight_path = input("Enter the ocr weight path: ")
    file_path = input("Enter the path to save the configuration file: ")
    create_weight_config(
        plate_weight_path=plate_weight_path,
        ocr_weight_path=ocr_weight_path,
        file_path=file_path
    )
    
    input("This script will generate a configuration file for Minio. Press Enter to continue...")
    endpoint = input("Enter the Minio endpoint: ")
    access_key = input("Enter the access key: ")
    secret_key = input("Enter the secret key: ")
    file_path = input("Enter the path to save the configuration file: ")
    create_minio_config(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        file_path=file_path
    )    
