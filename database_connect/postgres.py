import psycopg2
from psycopg2 import sql
from datetime import datetime
import json
import generate_config.generate_config as config


# Database connection parameters
def get_postgres_config():
    global DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    config_data = config.read_postgres_config("config/postgres_config.ini")
    DB_HOST = config_data['host']
    DB_PORT = config_data['port']
    DB_NAME = config_data['database']
    DB_USER = config_data['user']
    DB_PASSWORD = config_data['password']
    
    

def update_heartbeat(timestamp, edge_device_id):
    try:
        # Connect to PostgreSQL
        get_postgres_config()
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = connection.cursor()
        # Update data
        update_query = """
        UPDATE "EdgeDevice"
        SET "onlineTime" = %s
        WHERE id = %s;
        """
        cursor.execute(update_query, (timestamp, edge_device_id))
        connection.commit()
        print("Data updated successfully.")

    except (Exception, psycopg2.Error) as error:
        print("Error while interacting with PostgreSQL:", error)

    finally:
        # Close the connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection closed.")
            
def insert_incident_log(object_id, class_id, confidence, marker_id, first_seen, last_seen, event, bbox, id_rule_applied):
    inserted_id = -1
    try:
        # Connect to PostgreSQL
        get_postgres_config()
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = connection.cursor()
        select_query = "SELECT \"edgeDeviceId\" FROM \"Marker\" WHERE id = " + str(marker_id) + ";"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        edge_device_id = rows[0][0]
        print("Edge Device ID: ", edge_device_id)

        
        select_query = "SELECT \"ruleId\" FROM \"RuleApplied\" WHERE id = " + str(id_rule_applied) + ";"
        cursor.execute(select_query)
        rows = cursor.fetchall()
        id_rule = rows[0][0]
        print("Rule ID: ", id_rule)

        name = "Incident " + event
        desc = "Incident " + event + " detected" + " with confidence " + str(confidence) + " at marker " + str(marker_id)
        created_at = datetime.now()
        
        select_edge_device_query = "SELECT \"projectId\" FROM \"EdgeDevice\" WHERE id = " + str(edge_device_id) + ";"
        cursor.execute(select_edge_device_query)
        rows = cursor.fetchall()
        project_id = rows[0][0]
        print("Project ID: ", project_id)
        
        json_info = json.dumps({
            "objectId": object_id, 
            "classId": class_id, 
            "confidence": confidence, 
            "marker_id": marker_id,
            "first_seen": first_seen, 
            "last_seen": last_seen, 
            "event": event, 
            "bbox": bbox, 
            "id_rule_applied": id_rule_applied})
        
        insert_query = """
        INSERT INTO "Incident" (name, "desc", "createdAt", "ruleAppliedId", "markerId", "edgeDeviceId", "projectId", "ruleId", "jsonInfo", confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        data = (name, desc, created_at,id_rule_applied, marker_id, edge_device_id, project_id, id_rule, json_info, confidence)
        cursor.execute(insert_query,data)
        print("Data inserted successfully.")
        inserted_id = cursor.fetchone()[0]
        connection.commit()
        print("get inserted id: ", inserted_id)
        
    except (Exception, psycopg2.Error) as error:
        print("Error while interacting with PostgreSQL:", error)

    finally:
        # Close the connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection closed.")
        return inserted_id
            
def update_incident(image_path, incident_id, license_plate_number, license_plate_province):
    try:
        # Connect to PostgreSQL
        get_postgres_config()
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = connection.cursor()
        image_path = "/api/download/minio?object="+ image_path +"&bucket=incident-image"
        # Update data
        update_query = """
        UPDATE "Incident"
        SET 
        "imageSource" = %s,
        "lincensePlateNumber" = %s,
        "lincensePlateProvince" = %s
        WHERE id = %s;
        """
        cursor.execute(update_query, (image_path ,license_plate_number, license_plate_province, incident_id))
        connection.commit()
        print("Data updated successfully.")

    except (Exception, psycopg2.Error) as error:
        print("Error while interacting with PostgreSQL:", error)

    finally:
        # Close the connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection closed.")

def get_nvr_link_by_marker_id(marker_id):
    try:
        # Connect to PostgreSQL
        get_postgres_config()
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = connection.cursor()
        # Update data
        get_link_query = """
            SELECT ed.\"nvrUrl\"
            FROM \"Marker\" m
            JOIN \"EdgeDevice\" ed ON m.\"edgeDeviceId\" = ed.id
            WHERE m.id = %s;
        """
        cursor.execute(get_link_query, (marker_id,))
        result = cursor.fetchone()
        
        if result is not None:
            return result[0]
        else:
            return None
        
    except (Exception, psycopg2.Error) as error:
        print("Error while interacting with PostgreSQL:", error)

    finally:
        # Close the connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection closed.")
        
        
