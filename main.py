import os
import uuid
import mysql.connector
from mysql.connector import Error
from fastapi import FastAPI, HTTPException, Query, Depends
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
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from langchain_groq import ChatGroq
from langflow.components.tools import RetrieverToolComponent
from dotenv import load_dotenv
from typing import Callable, Union, Optional
from pydantic import BaseModel
from memory import save_query_and_answer, connect_to_db
from heading_generator import generate_heading
from utils import (validate_mongodb_uri, validate_mongodb_database_name, validate_mongodb_collection_name, 
                   validate_sql_connection,validate_mongodb_connection, create_enhanced_agent_prompt,get_current_user,
                   save_user_session, validate_sqlserver_connection
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
username = os.getenv("username")

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

class UserSession(BaseModel):
    email_id: str
    token: str
    role: str

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
            
            retriever_tool = RetrieverToolComponent()
            retriever = retriever_tool.build(
                retriever=toolkit,
                name="SQLRetriever",
                description="Retrieves data from SQL database"
            )
            
            prompt = create_enhanced_agent_prompt("sql")
            agent = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                retriever=retriever,
                verbose=verbose,
                handle_parsing_errors=True,
                prefix=prompt
            )

            return agent
        except Exception as e:
            raise Exception(f"Failed to connect to SQL database. Please check the database URI. Error: {str(e)}")

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
        
        # embeddings = GoogleGenerativeAIEmbeddings(
        #     model="models/embedding-001", 
        #     google_api_key=google_api_key,
        # )
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
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
                formatted_results = []
                for doc in results:
                    # Convert ObjectId to string and format the document
                    doc['_id'] = str(doc['_id'])
                    formatted_results.append(str(doc))
                return "\n".join(formatted_results)
            except Exception as e:
                return f"Error executing query: {str(e)}"

        tools = [
            Tool(
                name="MongoDB Query",
                func=query_mongodb,
                description="Useful for querying MongoDB database. Returns formatted results from the collection."
            )
        ]
        
        # Use a retriever tool for enhanced data retrieval
        retriever_tool = RetrieverToolComponent()
        retriever = retriever_tool.build(
            retriever=vector_store,
            name="NoSQLRetriever",
            description="Retrieves data from NoSQL database"
        )

        # Create MongoDB agent with enhanced prompting
        prompt = create_enhanced_agent_prompt("nosql")
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            agent_kwargs={"prefix": prompt}
        )
        
        return agent



def enhance_agent_configuration(agent_config: dict, db_type: str) -> dict:
    """Enhances the agent configuration with better prompting and response handling"""

    agent_config["system_message"] = create_enhanced_agent_prompt(db_type)
    
    agent_config.update({
        "response_format": {
            "type": "structured_natural_language",
            "include_metadata": True,
            "error_handling": "verbose"
        },
        "output_parser_config": {
            "require_complete_sentences": True,
            "maintain_context": True,
            "format_lists": True
        }
    })
    
    return agent_config


def create_sql_query_agent(api_key: str, database_uri: str, verbose: bool = False):
    # llm = ChatGoogleGenerativeAI(
    #     api_key=api_key,
    #     model="gemini-1.5-pro",
    #     temperature=0.5
    # )
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-70b-versatile",
        temperature=0.2
    )

    sql_agent_component = SQLAgentComponent()
    sql_agent = sql_agent_component.build(
        llm=llm,
        database_uri=database_uri,
        verbose=verbose
    )
    
    return sql_agent, sql_agent_component

def create_nosql_query_agent(
    # google_api_key: str,
    groq_api_key: str,
    mongodb_uri: str,
    db_name: str,
    collection_name: str,
    index_name: str = "default",
    verbose: bool = False
):
    # llm = ChatGoogleGenerativeAI(
    #     api_key=google_api_key,
    #     model="gemini-1.5-pro",
    #     temperature=0.5
    # )
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-70b-versatile",  # or any other Groq model you prefer
        temperature=0.2
    )

    
    nosql_agent_component = NoSQLAgentComponent()
    nosql_agent = nosql_agent_component.build(
        llm=llm,
        mongodb_uri=mongodb_uri,
        db_name=db_name,
        collection_name=collection_name,
        # google_api_key=google_api_key,
        google_api_key=groq_api_key,

        index_name=index_name,
        verbose=verbose
    )
    
    return nosql_agent, nosql_agent_component

def query_database(agent: AgentExecutor, question: str) -> str:
    try:
        enhanced_question = f"""
        Please provide a concise and clear response to the following question,
        keeping the answer focused and to the point. If the requested information is not available,
        explain what was searched for and suggest relevant alternatives.

        Question: {question}
        """
        response = agent.invoke({"input": enhanced_question})
        
        output = response["output"].strip()
        if "Final Answer:" in output:
            formatted_response = output.split("Final Answer:")[-1].strip()
        else:
            formatted_response = output
        
        if len(formatted_response.split()) < 10:
            formatted_response = f"""Based on the database search: {formatted_response}

If you'd like more specific information, please feel free to ask about particular tables or data points."""
            
        return formatted_response
        
    except Exception as e:
        return f"Error querying database: {str(e)}"


@app.post("/connect-db")
async def connect_db(request: QueryRequest):
    try:
        if request.db_type not in ["sql", "nosql"]:
            raise HTTPException(
                status_code=400,
                detail="db_type must be either 'sql' or 'nosql'"
            )

        if request.db_type == "sql":
            valid_prefixes = ['postgresql://', 'mysql://', 'mssql+pyodbc://']
            if not any(prefix in request.database_uri.lower() for prefix in valid_prefixes):
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid SQL URI format. Must be PostgreSQL, MySQL, or SQL Server URI."
                )

            is_valid_sql, sql_error = validate_sql_connection(request.database_uri)
            if not is_valid_sql:
                raise HTTPException(status_code=400, detail=sql_error)

            if request.db_name or request.collection_name:
                raise HTTPException(
                    status_code=400,
                    detail="db_name and collection_name should not be provided for SQL connections"
                )
        else:
            # Validate MongoDB connection
            is_valid_uri, uri_error = validate_mongodb_uri(request.database_uri)
            if not is_valid_uri:
                raise HTTPException(status_code=400, detail=uri_error)

            if not request.db_name or not request.collection_name:
                raise HTTPException(
                    status_code=400,
                    detail="db_name and collection_name are required for MongoDB connections"
                )

            is_valid_db, db_error = validate_mongodb_database_name(request.db_name)
            if not is_valid_db:
                raise HTTPException(status_code=400, detail=db_error)

            is_valid_collection, collection_error = validate_mongodb_collection_name(request.collection_name)
            if not is_valid_collection:
                raise HTTPException(status_code=400, detail=collection_error)

            is_valid_connection, connection_error = validate_mongodb_connection(
                request.database_uri, 
                request.db_name, 
                request.collection_name
            )
            if not is_valid_connection:
                raise HTTPException(status_code=400, detail=connection_error)

        # Save to database
        with connect_to_db() as connection:
            with connection.cursor(dictionary=True) as cursor:
                # Check existing record
                cursor.execute("SELECT * FROM db_connections LIMIT 1")
                existing_record = cursor.fetchone()

                while cursor.nextset():
                    pass

                if existing_record:
                    # Update existing record
                    update_query = """
                    UPDATE db_connections 
                    SET database_uri = %s, 
                        db_type = %s, 
                        db_name = %s, 
                        collection_name = %s, 
                        allowed_databases = %s 
                    WHERE id = %s
                    """
                    
                    if request.db_type == "sql":
                        allowed_dbs = 'mssql' if 'mssql+pyodbc://' in request.database_uri.lower() else 'mysql,postgresql'
                        params = (
                            request.database_uri,
                            request.db_type,
                            None,
                            None,
                            allowed_dbs,
                            existing_record['id']
                        )
                    else:
                        params = (
                            request.database_uri,
                            request.db_type,
                            request.db_name,
                            request.collection_name,
                            'mongodb',
                            existing_record['id']
                        )
                    
                    cursor.execute(update_query, params)
                    message = "Database connection updated successfully!"
                else:
                    # Insert new record
                    insert_query = """
                    INSERT INTO db_connections 
                    (database_uri, db_type, db_name, collection_name, allowed_databases)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    
                    if request.db_type == "sql":
                        allowed_dbs = 'mssql' if 'mssql+pyodbc://' in request.database_uri.lower() else 'mysql,postgresql'
                        params = (
                            request.database_uri,
                            request.db_type,
                            None,
                            None,
                            allowed_dbs
                        )
                    else:
                        params = (
                            request.database_uri,
                            request.db_type,
                            request.db_name,
                            request.collection_name,
                            'mongodb'
                        )
                    
                    cursor.execute(insert_query, params)
                    message = "Database connection saved successfully!"

                connection.commit()

        return {"mesage": message}

    except mysql.connector.Error as db_err:
        raise HTTPException(status_code=500, detail=f"Database error: {str(db_err)}")
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/chat")
async def chat(
    request: QueryRequest,
    thread_id: str = Query(None),
    current_user: UserSession = Depends(get_current_user)
):
    try:
        connection = connect_to_db()
        if not connection:
            raise HTTPException(status_code=500, detail="Unable to connect to the database.")

        cursor = connection.cursor(dictionary=True)
        
        email_id = current_user.email_id
        token = current_user.token
        role = current_user.role

        await save_user_session(email_id,role, token, connection)
        
        cursor.execute("SELECT * FROM db_connections ORDER BY created_at DESC LIMIT 1")
        db_connection = cursor.fetchone()

        if not db_connection:
            raise HTTPException(status_code=400, detail="No database found. Connect to the database first.")
        
        database_uri = db_connection["database_uri"]
        db_type = db_connection["db_type"]
        db_name = db_connection["db_name"]
        collection_name = db_connection["collection_name"]
        allowed_databases = db_connection["allowed_databases"].split(',')
        
        if db_type == "sql":
            if not any(db in database_uri.lower() for db in allowed_databases):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid SQL database URI. Supported databases: {', '.join(allowed_databases)}."
                )
        elif db_type == "nosql":
            if not any(db in database_uri.lower() for db in allowed_databases):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid NoSQL database URI. Supported databases: {', '.join(allowed_databases)}."
                )

        is_new_thread = thread_id is None
        if is_new_thread:
            thread_id = str(uuid.uuid4())

        question = request.question
        # google_api_key = GOOGLE_API_KEY
        groq_api_key = GROQ_API_KEY
        
        if db_type == "sql":
            sql_agent, component = create_sql_query_agent(
                api_key=groq_api_key,
                database_uri=database_uri,
                verbose=True
            )
        elif db_type == "nosql":
            mongodb_uri = database_uri
            sql_agent, component = create_nosql_query_agent(
                api_key=groq_api_key,
                mongodb_uri=mongodb_uri,
                db_name=db_name,
                collection_name=collection_name,
                verbose=True
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid database type specified")

        response = query_database(sql_agent, question)

        heading = None
        query_heading = """
            SELECT heading FROM history 
            WHERE thread_id = %s AND email_id = %s 
            LIMIT 1
        """
        cursor.execute(query_heading, (thread_id, current_user.email_id))
        result = cursor.fetchone()

        if result and result.get("heading"):
            heading = result["heading"]
        elif is_new_thread:
            heading = await generate_heading(request.question, response)

        save_query = """
            INSERT INTO history (thread_id, user_query, ai_response, heading, email_id)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(save_query, (
            thread_id,
            question,
            response,
            heading,
            current_user.email_id
        ))
        connection.commit()

        query_history = """
            SELECT * FROM history 
            WHERE thread_id = %s AND email_id = %s 
            ORDER BY id ASC
        """
        cursor.execute(query_history, (thread_id, current_user.email_id))
        history = cursor.fetchall()

        cursor.close()
        connection.close()

        # Format response
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
            "email_id": current_user.email_id,
            "role": role
        }

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/chat")
async def get_history(
    user_session: UserSession = Depends(get_current_user), 
    thread_id: Optional[str] = None
):
    try:
        email_id = user_session.email_id
        role = user_session.role
        # Directly use the username from .env
        username = os.getenv("username")

        connection = connect_to_db()
        if connection:
            cursor = connection.cursor(dictionary=True)

            if thread_id:
                query = "SELECT * FROM history WHERE thread_id = %s AND email_id = %s ORDER BY id ASC"
                cursor.execute(query, (thread_id, email_id))
            else:
                query = "SELECT * FROM history WHERE email_id = %s ORDER BY id DESC"
                cursor.execute(query, (email_id,))

            history = cursor.fetchall()
            cursor.close()
            connection.close()

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

            return [{"role": role, "username": username, **item} for item in grouped_history.values()]

        else:
            raise HTTPException(status_code=500, detail="Database connection failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")



if __name__ == "__main__":
    pass
