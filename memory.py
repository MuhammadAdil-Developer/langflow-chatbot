import mysql.connector
from mysql.connector import Error

# MySQL Database Connection Settings
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'admin',
    'password': 'admin123',
    'database': 'MyDB'
}

def connect_to_db():
    """Connect to the MySQL database and return the connection object."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error: {e}")
        return None

def save_query_and_answer(user_query: str, ai_response: str, thread_id: str, heading: str = None):
    try:
        connection = connect_to_db()
        if connection:
            cursor = connection.cursor()
            # Insert query and answer into the history table, including the heading
            query = """
                INSERT INTO history (thread_id, user_query, ai_response, heading)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (thread_id, user_query, ai_response, heading))
            connection.commit()
            cursor.close()
            connection.close()
    except Error as e:
        print(f"Error saving query and answer: {str(e)}")

