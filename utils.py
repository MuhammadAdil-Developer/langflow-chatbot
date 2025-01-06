import re
from urllib.parse import urlparse
from typing import Optional, Tuple
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import requests
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import mysql.connector
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import os

# Load environment variables
load_dotenv()

AUTH_TOKEN_URL = os.getenv("AUTH_TOKEN_URL")
USER_LOGIN_URL = os.getenv("USER_LOGIN_URL")
AUTH_CODE = os.getenv("AUTH_CODE")
username = os.getenv("username")
PASSWORD = os.getenv("PASSWORD")
GRANT_TYPE = os.getenv("GRANT_TYPE")


class UserSession(BaseModel):
    email_id: str
    role: str
    token: str

async def get_auth_token() -> str:
    """
    Fetches an authentication token from the AUTH_TOKEN_URL.
    """
    payload = {
        "grant_type": GRANT_TYPE,
        "AuthCode": AUTH_CODE
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    try:
        response = requests.post(AUTH_TOKEN_URL, data=payload, headers=headers)
        response.raise_for_status()

        token_data = response.json()
        token = token_data.get("access_token") or token_data.get("token")

        if not token:
            raise HTTPException(
                status_code=500,
                detail="Token not found in response."
            )
        return token
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch auth token: {str(e)}"
        )


async def save_user_session(email_id: str, role: str, token: str, connection):
    """
    Saves the user's session details into the database.
    """
    query = """
    INSERT INTO user_sessions (email_id, token, last_login) 
    VALUES (%s, %s, CURRENT_TIMESTAMP)
    ON DUPLICATE KEY UPDATE 
        token = VALUES(token),
        last_login = CURRENT_TIMESTAMP
    """
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, (email_id, token))
        connection.commit()
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    finally:
        cursor.close()



async def get_user_details(token: str) -> UserSession:
    """
    Retrieves user details using the provided token.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "userName": username,
        "password": PASSWORD
    }

    try:
        response = requests.post(USER_LOGIN_URL, headers=headers, json=payload)
        response.raise_for_status()

        if response.status_code == 200:
            data = response.json()
            email_id = data.get("EmailID")
            role = data.get("Role")

            if not email_id:
                raise HTTPException(
                    status_code=500,
                    detail="EmailID not found in response."
                )
            if not role:
                raise HTTPException(
                    status_code=500,
                    detail="Role not found in response."
                )

            return UserSession(email_id=email_id, token=token, role=role)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve user details: {response.status_code} - {response.text}"
            )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user details: {str(e)}"
        )


async def get_current_user(authorization: Optional[str] = Header(None)) -> UserSession:
    """
    Automatically retrieves the token and fetches the user email_id.
    """
    try:
        token = await get_auth_token() if not authorization else authorization.replace("Bearer ", "").strip()

        if not token:
            raise HTTPException(status_code=401, detail="Invalid or missing token.")
        
        user_session = await get_user_details(token)
        return user_session
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


def validate_mongodb_uri(uri: str) -> Tuple[bool, Optional[str]]:
    """
    Validates MongoDB URI format and components.
    Returns (is_valid, error_message)
    """
    try:
        if not uri.startswith(('mongodb://', 'mongodb+srv://')):
            return False, "MongoDB URI must start with 'mongodb://' or 'mongodb+srv://'"

        parsed = urlparse(uri)
        
        if 'localhost' not in parsed.netloc and '127.0.0.1' not in parsed.netloc:
            if '@' not in parsed.netloc:
                return False, "Remote MongoDB connections must include username and password"
            
            credentials = parsed.netloc.split('@')[0]
            if ':' not in credentials:
                return False, "MongoDB credentials must include both username and password"

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
        client.server_info()
        
        if db_name not in client.list_database_names():
            return False, f"Database '{db_name}' does not exist."

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
    try:
        if 'mssql+pyodbc://' in uri.lower():
            return True, None  # Skip validation for now, just allow saving
        engine = create_engine(uri)
        with engine.connect() as connection:
            return True, None
    except Exception as e:
        return False, f"Database connection error: {str(e)}"


def validate_sqlserver_connection(database_uri):
    connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=MyDatabase;UID=newuser;PWD=Adil59999"
    try:
        import pyodbc
        conn = pyodbc.connect(connection_string)
        print("Connection successful!")
        conn.close()
    except Exception as e:
        print(f"Connection failed: {e}")



def create_enhanced_agent_prompt(db_type: str) -> str:
    """Creates an enhanced prompt for database agents to provide better structured responses"""
    
    if db_type == "sql":
        return """You are an intelligent SQL database assistant. Please provide concise and clear answers to user queries.

When answering queries:
- Keep responses short and to the point
- Provide the necessary context and explanation without over-explaining
- If information isn't found, explain briefly what was searched and suggest alternatives
- Avoid unnecessary details, but make sure the response is complete and understandable

Remember to:
- Maintain proper grammar and complete sentences
- Avoid excessive verbosity
- Provide examples only when necessary"""

    else:  # nosql
        return """You are an intelligent MongoDB database assistant. Please provide concise and clear answers to user queries.

When answering queries:
- Keep responses short and to the point
- Provide the necessary context and explanation without over-explaining
- If information isn't found, explain briefly what was searched and suggest alternatives
- Avoid unnecessary details, but make sure the response is complete and understandable

Remember to:
- Maintain proper grammar and complete sentences
- Avoid excessive verbosity
- Provide examples only when necessary"""
