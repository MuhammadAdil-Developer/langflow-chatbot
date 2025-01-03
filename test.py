import uuid
import mysql.connector
from mysql.connector import Error
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langflow.custom import CustomComponent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langflow.field_typing import LanguageModel
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools import Tool
from pymongo import MongoClient
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Callable, Union
from pydantic import BaseModel
from memory import save_query_and_answer, connect_to_db
from heading_generator import generate_heading
from utils import validate_mongodb_uri, validate_mongodb_database_name, validate_mongodb_collection_name, validate_sql_connection,validate_mongodb_connection

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Request Model for handling input
class QueryRequest(BaseModel):
    question: str = None
    db_name: str = None  
    database_uri: str = None  
    collection_name: str = None
    db_type: str = None

# SQL Agent Component
class SQLAgentComponent:
    display_name = "SQLAgent"
    description = "Construct an SQL agent from an LLM and tools."

    def build_config(self):
        return {
            "llm": {"display_name": "LLM"},
            "database_uri": {"display_name": "Database URI"},
            "verbose": {"display_name": "Verbose", "value": False, "advanced": True},
        }

    def build(self, llm, database_uri: str, verbose: bool = False):
        try:
            db = SQLDatabase.from_uri(database_uri)
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            return create_sql_agent(llm=llm, toolkit=toolkit, verbose=verbose, handle_parsing_errors=True)
        except Exception as e:
            raise Exception(f"Failed to connect to SQL database. Please check the database URI. Error: {str(e)}")


# NoSQL Agent Component (MongoDB)
class NoSQLAgentComponent(CustomComponent):
    display_name = "NoSQLAgent"
    description = "Construct a NoSQL agent from an LLM and tools."

    def build_config(self):
        return {
            "llm": {"display_name": "LLM"},
            "mongodb_uri": {"display_name": "MongoDB URI"},
            "db_name": {"display_name": "Database Name"},
            "collection_name": {"display_name": "Collection Name"},
            "index_name": {"display_name": "Index Name", "value": "default"},
            "verbose": {"display_name": "Verbose", "value": False, "advanced": True},
        }

    def build(
        self,
        llm: LanguageModel,
        mongodb_uri: str,
        db_name: str,
        collection_name: str,
        google_api_key: str,
        index_name: str = "default",
        verbose: bool = False,
    ) -> Union[AgentExecutor, Callable]:
        client = MongoClient(mongodb_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=google_api_key,
        )
        
        vector_store = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=mongodb_uri,
            namespace=f"{db_name}.{collection_name}",
            embedding=embeddings,
            index_name=index_name
        )
        
        def query_mongodb(query_str: str) -> str:
            try:
                results = collection.find().sort('_id', -1).limit(5)
                return "\n".join([str(doc) for doc in results])
            except Exception as e:
                return f"Error executing query: {str(e)}"

        tools = [
            Tool(
                name="MongoDB Query",
                func=query_mongodb,
                description="Useful for querying MongoDB database."
            )
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose
        )
        
        return agent

# Function to create SQL Agent
def create_sql_query_agent(api_key: str, database_uri: str, verbose: bool = False):
    llm = ChatGoogleGenerativeAI(
        api_key="AIzaSyAEOVGaFTlqhwDVhuw5XuddBOM6ZNoYIdk",
        model="gemini-1.5-flash",
        temperature=0
    )

    sql_agent_component = SQLAgentComponent()
    sql_agent = sql_agent_component.build(
        llm=llm,
        database_uri=database_uri,
        verbose=verbose
    )

    return sql_agent, sql_agent_component

# Function to create NoSQL Agent
def create_nosql_query_agent(
    google_api_key: str,
    mongodb_uri: str,
    db_name: str,
    collection_name: str,
    index_name: str = "default",
    verbose: bool = False
):
    llm = ChatGoogleGenerativeAI(
        api_key=google_api_key,
        model="gemini-1.5-pro",
        temperature=0
    )
    
    nosql_agent_component = NoSQLAgentComponent()
    
    nosql_agent = nosql_agent_component.build(
        llm=llm,
        mongodb_uri=mongodb_uri,
        db_name=db_name,
        collection_name=collection_name,
        google_api_key=google_api_key,
        index_name=index_name,
        verbose=verbose
    )
    
    return nosql_agent, nosql_agent_component

# Function to query the database using the agent
def query_database(agent: AgentExecutor, question: str) -> str:
    try:
        response = agent.invoke({"input": question})
        return response["output"]
    except Exception as e:
        return f"Error querying database: {str(e)}"


@app.post("/connect-db")
async def connect_db(request: QueryRequest):
    try:
        # Validate database type
        if request.db_type not in ["sql", "nosql"]:
            raise HTTPException(
                status_code=400,
                detail="db_type must be either 'sql' or 'nosql'"
            )

        if request.db_type == "sql":
            # SQL validation
            if not any(prefix in request.database_uri.lower() for prefix in ['postgresql://', 'mysql://']):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid SQL database URI. Must be PostgreSQL or MySQL URI format."
                )
            
            # Validate SQL connection
            is_valid_sql, sql_error = validate_sql_connection(request.database_uri)
            if not is_valid_sql:
                raise HTTPException(status_code=400, detail=sql_error)

            # Ensure db_name and collection_name are not provided for SQL
            if request.db_name or request.collection_name:
                raise HTTPException(
                    status_code=400,
                    detail="db_name and collection_name should not be provided for SQL connections"
                )
        else: 
            # MongoDB validation
            is_valid_uri, uri_error = validate_mongodb_uri(request.database_uri)
            if not is_valid_uri:
                raise HTTPException(status_code=400, detail=uri_error)

            # Validate MongoDB database and collection
            is_valid_connection, connection_error = validate_mongodb_connection(request.database_uri, request.db_name, request.collection_name)
            if not is_valid_connection:
                raise HTTPException(status_code=400, detail=connection_error)

            # Database name validation
            is_valid_db, db_error = validate_mongodb_database_name(request.db_name)
            if not is_valid_db:
                raise HTTPException(status_code=400, detail=db_error)

            # Collection name validation
            is_valid_collection, collection_error = validate_mongodb_collection_name(request.collection_name)
            if not is_valid_collection:
                raise HTTPException(status_code=400, detail=collection_error)

        # Proceed with saving the connection details to the database
        with connect_to_db() as connection:
            with connection.cursor(dictionary=True) as cursor:
                # Check if any record exists
                cursor.execute("SELECT * FROM db_connections LIMIT 1")
                existing_record = cursor.fetchone()
                
                while cursor.nextset():
                    pass

                if existing_record:
                    # Update existing record
                    update_query = "UPDATE db_connections SET database_uri = %s, db_type = %s"
                    update_params = [request.database_uri, request.db_type]
                    
                    if request.db_type == "nosql":
                        update_query += ", db_name = %s, collection_name = %s, allowed_databases = %s"
                        update_params.extend([request.db_name, request.collection_name, 'mongodb'])
                    else:
                        update_query += ", db_name = NULL, collection_name = NULL, allowed_databases = %s"
                        update_params.append('mysql,postgresql')
                    
                    update_query += " WHERE id = %s"  # Assuming there is an 'id' column to identify records
                    update_params.append(existing_record['id'])
                    
                    cursor.execute(update_query, update_params)
                    message = "Database Connected!"
                else:
                    # Insert new record
                    insert_query = """
                        INSERT INTO db_connections 
                        (database_uri, db_type, db_name, collection_name, allowed_databases)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    if request.db_type == "nosql":
                        cursor.execute(insert_query, (
                            request.database_uri, 
                            request.db_type, 
                            request.db_name, 
                            request.collection_name, 
                            'mongodb'
                        ))
                    else:
                        cursor.execute(insert_query, (
                            request.database_uri, 
                            request.db_type, 
                            None, 
                            None, 
                            'mysql,postgresql'
                        ))
                    message = "Connection saved successfully."

                connection.commit()

        return {"message": message}

    except mysql.connector.Error as db_err:
        raise HTTPException(status_code=500, detail=f"Database error: {str(db_err)}")
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")



@app.post("/chat")
async def chat(request: QueryRequest, thread_id: str = Query(None)):
    try:
        connection = connect_to_db()
        if connection:
            cursor = connection.cursor(dictionary=True)
            
            # Fetch the most recent database connection
            cursor.execute("SELECT * FROM db_connections ORDER BY created_at DESC LIMIT 1")
            db_connection = cursor.fetchone()

            if not db_connection:
                raise HTTPException(status_code=400, detail="No database found. Connect to the database first.")
            
            # Retrieve connection details
            database_uri = db_connection["database_uri"]
            db_type = db_connection["db_type"]
            db_name = db_connection["db_name"]
            collection_name = db_connection["collection_name"]
            allowed_databases = db_connection["allowed_databases"].split(',')
            
            cursor.close()
            connection.close()
        else:
            raise HTTPException(status_code=500, detail="Unable to connect to the database.")

        # Validate database URI and type dynamically
        if db_type == "sql":
            if not any(db in database_uri.lower() for db in allowed_databases):
                raise HTTPException(status_code=400, detail=f"Invalid SQL database URI. Supported databases: {', '.join(allowed_databases)}.")
        elif db_type == "nosql":
            if not any(db in database_uri.lower() for db in allowed_databases):
                raise HTTPException(status_code=400, detail=f"Invalid NoSQL database URI. Supported databases: {', '.join(allowed_databases)}.")

        is_new_thread = thread_id is None
        if is_new_thread:
            thread_id = str(uuid.uuid4())

        question = request.question
        google_api_key = "AIzaSyAEOVGaFTlqhwDVhuw5XuddBOM6ZNoYIdk"

        if db_type == "sql":
            sql_agent, component = create_sql_query_agent(
                api_key=google_api_key,
                database_uri=database_uri,
                verbose=True
            )
        elif db_type == "nosql":
            mongodb_uri = database_uri
            # Use dynamically fetched db_name and collection_name
            sql_agent, component = create_nosql_query_agent(
                google_api_key=google_api_key,
                mongodb_uri=mongodb_uri,
                db_name=db_name,
                collection_name=collection_name,
                verbose=True
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid database type specified")

        response = query_database(sql_agent, question)

        heading = None
        connection = connect_to_db()
        if connection:
            cursor = connection.cursor(dictionary=True)

            query_heading = "SELECT heading FROM history WHERE thread_id = %s LIMIT 1"
            cursor.execute(query_heading, (thread_id,))
            result = cursor.fetchone()

            if result and result.get("heading"):
                heading = result["heading"]
            elif is_new_thread:
                user_message = request.question
                ai_response = response 
                heading = await generate_heading(user_message, ai_response)

            cursor.close()
            connection.close()

        save_query_and_answer(question, response, thread_id, heading)

        connection = connect_to_db()
        if connection:
            cursor = connection.cursor(dictionary=True)
            query_history = "SELECT * FROM history WHERE thread_id = %s ORDER BY id ASC"
            cursor.execute(query_history, (thread_id,))
            history = cursor.fetchall()
            cursor.close()
            connection.close()

        human_messages = []
        ai_responses = []

        for record in history:
            human_messages.append(record["user_query"])
            ai_responses.append(record["ai_response"])

        response_data = {
            "thread_id": thread_id,
            "ai_response": ai_responses,
            "human_message": human_messages,
            "heading": heading,
        }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/chat")
async def get_history(thread_id: str = None):
    try:
        connection = connect_to_db()
        if connection:
            cursor = connection.cursor(dictionary=True)

            # SQL query to fetch all history or filter by thread_id
            if thread_id:
                # For a specific thread, fetch in normal order (oldest first)
                query = "SELECT * FROM history WHERE thread_id = %s ORDER BY id ASC"
                cursor.execute(query, (thread_id,))
            else:
                # For all history, get the latest data first (descending order by id or timestamp)
                query = "SELECT * FROM history ORDER BY id DESC"
                cursor.execute(query)

            history = cursor.fetchall()
            cursor.close()
            connection.close()

            # Grouping messages and responses by thread_id
            grouped_history = {}
            for record in history:
                thread_id = record["thread_id"]
                if thread_id not in grouped_history:
                    grouped_history[thread_id] = {
                        "thread_id": thread_id,
                        "human_message": [],
                        "ai_response": [],
                        "heading": record["heading"],
                    }
                grouped_history[thread_id]["human_message"].append(record["user_query"])
                grouped_history[thread_id]["ai_response"].append(record["ai_response"])

            # Transforming grouped history into a list
            response = list(grouped_history.values())

            return response

        else:
            raise HTTPException(status_code=500, detail="Database connection failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


if __name__ == "__main__":
    pass
