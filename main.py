# from langflow.custom import CustomComponent
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_community.agent_toolkits.sql.base import create_sql_agent
# from langflow.field_typing import LanguageModel
# from langchain.agents import AgentExecutor
# from langchain_google_genai import ChatGoogleGenerativeAI

# from typing import Callable, Union
# # Replace this with the appropriate GenAI import if needed
# # from genai import GenAIModel  # Example import for GenAI


# class SQLAgentComponent(CustomComponent):
#     display_name = "SQLAgent"
#     description = "Construct an SQL agent from an LLM and tools."

#     def build_config(self):
#         return {
#             "llm": {"display_name": "LLM"},
#             "database_uri": {"display_name": "Database URI"},
#             "verbose": {"display_name": "Verbose", "value": False, "advanced": True},
#         }

#     def build(
#         self,
#         llm: LanguageModel,
#         database_uri: str,
#         verbose: bool = False,
#     ) -> Union[AgentExecutor, Callable]:
#         db = SQLDatabase.from_uri(database_uri)
#         toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#         return create_sql_agent(llm=llm, toolkit=toolkit, verbose=verbose, handle_parsing_errors=True)

# def create_sql_query_agent(api_key: str, database_uri: str, verbose: bool = False):
#     try:
#         # Replace with the appropriate GenAI initialization
#         llm = ChatGoogleGenerativeAI(
#             api_key="AIzaSyAEOVGaFTlqhwDVhuw5XuddBOM6ZNoYIdk",
#             model="gemini-1.5-flash",
#             temperature=0
#         )
        
#         sql_agent_component = SQLAgentComponent()
        
#         sql_agent = sql_agent_component.build(
#             llm=llm,
#             database_uri=database_uri,
#             verbose=verbose
#         )
        
#         return sql_agent, sql_agent_component
        
#     except Exception as e:
#         raise Exception(f"Error creating SQL agent: {str(e)}")

# def query_database(agent: AgentExecutor, question: str) -> str:
#     try:
#         response = agent.run(question)
#         return response
#     except Exception as e:
#         return f"Error querying database: {str(e)}"

# if __name__ == "__main__":
#     openai_api_key = "AIzaSyAEOVGaFTlqhwDVhuw5XuddBOM6ZNoYIdk"
#     database_uri = "postgresql://adiladmin:admin123@localhost:5432/chinook"
    
#     try:
#         sql_agent, component = create_sql_query_agent(
#             api_key=openai_api_key,
#             database_uri=database_uri,
#             verbose=True
#         )
        
#         questions = [# Note: Replace **<YOUR_APPLICATION_TOKEN>** with your actual Application token





from langflow.custom import CustomComponent
from langflow.field_typing import LanguageModel
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from typing import Callable, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools import Tool
from pymongo import MongoClient
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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
        try:
            # Initialize MongoDB client
            client = MongoClient(mongodb_uri)
            db = client[db_name]
            collection = db[collection_name]
            
            # Initialize Google embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",  # Google's embedding model
                google_api_key=google_api_key,
            )
            
            # Create vector store
            vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                connection_string=mongodb_uri,
                namespace=f"{db_name}.{collection_name}",
                embedding=embeddings,
                index_name=index_name
            )
            
            # Create MongoDB query tool
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
                    description="Useful for querying MongoDB database to find recent entries and specific information."
                )
            ]
            
            # Initialize agent
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=verbose
            )
            
            return agent
            
        except Exception as e:
            raise Exception(f"Error building NoSQL agent: {str(e)}")

def create_nosql_query_agent(
    google_api_key: str,
    mongodb_uri: str,
    db_name: str,
    collection_name: str,
    index_name: str = "default",
    verbose: bool = False
):
    try:
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
        
    except Exception as e:
        raise Exception(f"Error creating NoSQL agent: {str(e)}")

def query_database(agent: AgentExecutor, question: str) -> str:
    try:
        response = agent.invoke({"input": question})
        return response["output"]
    except Exception as e:
        return f"Error querying database: {str(e)}"

if __name__ == "__main__":
    # Configuration
    google_api_key = "AIzaSyAEOVGaFTlqhwDVhuw5XuddBOM6ZNoYIdk"
    mongodb_uri = "mongodb://localhost:27017/"
    db_name = "mydatabase"
    collection_name = "mycollection"
    index_name = "vector_index"
    
    try:
        # First, verify MongoDB connection
        client = MongoClient(mongodb_uri)
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        
        # Create and test the agent
        nosql_agent, component = create_nosql_query_agent(
            google_api_key=google_api_key,
            mongodb_uri=mongodb_uri,
            db_name=db_name,
            collection_name=collection_name,
            index_name=index_name,
            verbose=True
        )
        
        questions = [
            "create a level of effort with module and write in table format",
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            response = query_database(nosql_agent, question)
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()