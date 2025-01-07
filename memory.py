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

def save_query_and_answer(question: str, answer: str, thread_id: str = None):
    """Save the user query and agent's response into the MySQL database."""
    connection = connect_to_db()
    if connection is not None:
        cursor = connection.cursor()
        try:
            insert_query = """
            INSERT INTO history (user_query, ai_response, thread_id)
            VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (question, answer, thread_id))
            connection.commit()
        except Error as e:
            print(f"Error while saving to database: {e}")
        finally:
            cursor.close()
            connection.close()
            
def save_query_and_answer(question: str, answer: str):
    """Save the user query and agent's response into the MySQL database."""
    connection = connect_to_db()
    if connection is not None:
        cursor = connection.cursor()
        try:
            insert_query = """
            INSERT INTO history (user_query, ai_response)
            VALUES (%s, %s)
            """
            cursor.execute(insert_query, (question, answer))
            connection.commit()
            print(f"Query and answer saved: {question} -> {answer}")
        except Error as e:
            print(f"Error while saving to database: {e}")
        finally:
            cursor.close()
            connection.close()
    else:
        print("Failed to connect to the database.")
