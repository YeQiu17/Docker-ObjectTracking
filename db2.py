import numpy as np
import time
import logging
from azure.cosmos import CosmosClient, PartitionKey, exceptions

# Cosmos DB configuration
COSMOS_DB_ENDPOINT = 'https://occupancytrackerdb.documents.azure.com:443/'
COSMOS_DB_KEY = 'gC72Ce5BaUhA2oExqWEWZUsUjDlXqXfEO9mvZLVcGCEi5doy7hcuuBhWWIsDsTsKwrwE3Bg79uNXACDbpl2omA=='
DATABASE_NAME = 'Tracker'
CONTAINER_NAME = 'person'
PARTITION_KEY = '/person_id'

# Initialize the Cosmos client
cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = cosmos_client.get_database_client(DATABASE_NAME)
person_container = database.get_container_client(CONTAINER_NAME)

# Feature Vector Serialization/Deserialization
def get_cosmos_client():
    try:
        return CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
    except exceptions.CosmosHttpResponseError as e:
        logging.error(f"Failed to initialize Cosmos Client: {str(e)}")
        raise

cosmos_client = get_cosmos_client()
database = cosmos_client.get_database_client(DATABASE_NAME)
person_container = database.get_container_client(CONTAINER_NAME)

# Feature Vector Serialization/Deserialization
def serialize_feature_vector(vector):
    return vector.tolist()  # Convert NumPy array to list for JSON serialization

def deserialize_feature_vector(feature_list):
    return np.array(feature_list, dtype=np.float32)

def save_feature_to_db(obj_id, feature_vector):
    container = person_container
    serialized_feature_vector = serialize_feature_vector(feature_vector)
    item = {
        'id': str(obj_id),
        'feature_vector': serialized_feature_vector,
        'last_seen': time.time()  # Use real-time timestamp
    }
    try:
        container.upsert_item(item)
        logging.debug(f"Saved feature for object ID {obj_id} to Cosmos DB.")
    except exceptions.CosmosHttpResponseError as e:
        logging.error(f"Failed to save feature for object ID {obj_id}: {str(e)}")

def get_features_from_db():
    container = person_container
    try:
        query = "SELECT * FROM c"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        features = {item['id']: deserialize_feature_vector(item['feature_vector']) for item in items}
        logging.debug(f"Retrieved {len(features)} features from Cosmos DB.")
        return features
    except exceptions.CosmosHttpResponseError as e:
        logging.error(f"Failed to retrieve features: {str(e)}")
        return {}

def update_feature_in_db(obj_id, feature_vector):
    container = person_container
    serialized_feature_vector = serialize_feature_vector(feature_vector)
    try:
        item = container.read_item(item=str(obj_id), partition_key=str(obj_id))
        item['feature_vector'] = serialized_feature_vector
        item['last_seen'] = time.time()
        container.upsert_item(item)
        logging.debug(f"Updated feature for object ID {obj_id} in Cosmos DB.")
    except exceptions.CosmosResourceNotFoundError:
        logging.warning(f"Cannot update feature: ID {obj_id} not found in Cosmos DB.")
    except exceptions.CosmosHttpResponseError as e:
        logging.error(f"Failed to update feature for object ID {obj_id}: {str(e)}")

def delete_stale_features(timeout_threshold):
    container = person_container
    current_time = time.time()
    query = f"SELECT * FROM c WHERE c.last_seen < {current_time - timeout_threshold}"
    try:
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        for item in items:
            try:
                container.delete_item(item=item['id'], partition_key=item['id'])
                logging.debug(f"Deleted stale feature for object ID {item['id']} from Cosmos DB.")
            except exceptions.CosmosHttpResponseError as e:
                logging.error(f"Failed to delete feature for object ID {item['id']}: {str(e)}")
    except exceptions.CosmosHttpResponseError as e:
        logging.error(f"Failed to query stale features: {str(e)}")



def get_unique_people_count():
    try:
        query = "SELECT VALUE COUNT(1) FROM (SELECT DISTINCT c.id FROM c)"
        result = list(person_container.query_items(query=query, enable_cross_partition_query=True))
        unique_count = result[0] if result else 0
        logging.debug(f"Unique people count: {unique_count}")
        return unique_count
    except exceptions.CosmosHttpResponseError as e:
        logging.error(f"Failed to get unique people count: {str(e)}")
        return 0