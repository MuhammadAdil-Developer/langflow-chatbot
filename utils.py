import re
from urllib.parse import urlparse
from typing import Optional, Tuple
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


def validate_mongodb_uri(uri: str) -> Tuple[bool, Optional[str]]:
    """
    Validates MongoDB URI format and components.
    Returns (is_valid, error_message)
    """
    try:
        # Basic format check
        if not uri.startswith(('mongodb://', 'mongodb+srv://')):
            return False, "MongoDB URI must start with 'mongodb://' or 'mongodb+srv://'"

        # Parse the URI
        parsed = urlparse(uri)
        
        # Check for username and password if not localhost
        if 'localhost' not in parsed.netloc and '127.0.0.1' not in parsed.netloc:
            if '@' not in parsed.netloc:
                return False, "Remote MongoDB connections must include username and password"
            
            # Extract credentials
            credentials = parsed.netloc.split('@')[0]
            if ':' not in credentials:
                return False, "MongoDB credentials must include both username and password"

        # Check port if specified
        if ':' in parsed.netloc.split('@')[-1]:
            try:
                port = int(parsed.netloc.split(':')[-1])
                if not (0 <= port <= 65535):
                    return False, "Port number must be between 0 and 65535"
            except ValueError:
                return False, "Invalid port number"

        return True, None

    except Exception as e:
        return False, f"Invalid MongoDB URI format: {str(e)}"

def validate_mongodb_collection_name(collection_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validates MongoDB collection name according to MongoDB rules.
    Returns (is_valid, error_message)
    """
    if not collection_name:
        return False, "Collection name cannot be empty"
    
    if len(collection_name) > 255:
        return False, "Collection name cannot exceed 255 characters"
    
    # Check for invalid characters and patterns
    invalid_patterns = [
        (r'^system\.', "Collection name cannot start with 'system.'"),
        (r'\$', "Collection name cannot contain '$'"),
        (r'^$', "Collection name cannot be empty"),
        (r'\s', "Collection name cannot contain whitespace"),
        (r'\\', "Collection name cannot contain backslash"),
        (r'\x00', "Collection name cannot contain null characters")
    ]
    
    for pattern, message in invalid_patterns:
        if re.search(pattern, collection_name):
            return False, message
            
    return True, None

def validate_mongodb_database_name(db_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validates MongoDB database name according to MongoDB rules.
    Returns (is_valid, error_message)
    """
    if not db_name:
        return False, "Database name cannot be empty"
        
    if len(db_name) > 64:
        return False, "Database name cannot exceed 64 characters"
    
    # Check for invalid characters and patterns
    invalid_patterns = [
        (r'/', "Database name cannot contain '/'"),
        (r'\\', "Database name cannot contain '\\'"),
        (r'\s', "Database name cannot contain whitespace"),
        (r'\.', "Database name cannot contain '.'"),
        (r'^$', "Database name cannot be empty"),
        (r'\x00', "Database name cannot contain null characters")
    ]
    
    for pattern, message in invalid_patterns:
        if re.search(pattern, db_name):
            return False, message
            
    return True, None

def validate_mongodb_connection(uri: str, db_name: str, collection_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validates MongoDB connection URI and checks if the database and collection exist.
    Returns (is_valid, error_message)
    """
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Check if the server is reachable
        client.server_info()  # Will raise an exception if connection fails
        
        # Check if the specified database exists
        if db_name not in client.list_database_names():
            return False, f"Database '{db_name}' does not exist."
        
        # Check if the specified collection exists within the database
        db = client[db_name]
        if collection_name not in db.list_collection_names():
            return False, f"Collection '{collection_name}' does not exist in database '{db_name}'."
        
        return True, None
    except ServerSelectionTimeoutError as e:
        return False, f"Failed to connect to MongoDB: {str(e)}"
    except Exception as e:
        return False, f"Invalid MongoDB URI or error: {str(e)}"
    finally:
        client.close()


def validate_sql_connection(uri: str) -> Tuple[bool, Optional[str]]:
    """
    Validates SQL database connection URI and checks if the database is reachable.
    Returns (is_valid, error_message)
    """
    try:
        # Create SQLAlchemy engine to test the connection
        engine = create_engine(uri)
        with engine.connect() as connection:
            # If we reach here, the connection is valid
            return True, None
    except OperationalError as e:
        # Catch errors related to the SQL connection
        return False, f"Failed to connect to SQL database: {str(e)}"
    except Exception as e:
        return False, f"Invalid SQL database URI or error: {str(e)}"
