Overview

This application is built using FastAPI and allows users to connect to SQL and NoSQL databases, perform queries, and interact with a chat interface. It utilizes various components from Langchain to facilitate database interactions and responses.
Requirements
Before running the application, ensure you have the following installed:
Python 3.10 or higher
Required Python packages (listed below)
Required Packages

You can install the necessary packages using pip. Create a requirements.txt file with the following content:
text
fastapi
uvicorn
mysql-connector-python
pymongo
langchain
langflow
Then, install the packages using:
bash
pip install -r requirements.txt

Application Structure
The application consists of several key components:
FastAPI app: The main application that handles incoming requests.
Database Connection: Functions to connect to SQL (MySQL) and NoSQL (MongoDB) databases.
Agent Components: Components for creating SQL and NoSQL agents that handle queries.
User Session Management: Functions to manage user sessions and history.

Running the Application
Step 1: Setting Up Environment Variables
Before running the application, set up your environment variables for database connections and API keys. You can create a .env file in the root directory with the following content:
text

Step 2: Starting the Server
To run the FastAPI application, execute the following command in your terminal:
bash
uvicorn main:app --reload
Step 3: Accessing the API
Once the server is running, you can access the API at http://127.0.0.1:8000. You can also access the interactive API documentation at http://127.0.0.1:8000/docs.
API Endpoints
1. Connect to Database
POST /connect-db
Description: Connects to a specified SQL or NoSQL database.
Request Body:
json
{
  "question": "Your question here",
}
2. Chat Interface
POST /chat
Description: Sends a question to the connected database and retrieves a response.
Request Body:
json
{
  "question": "Your question here",
  "db_name": "Database name",
  "database_uri": "Database URI",
  "collection_name": "Collection name",
  "db_type": "sql or nosql"
}
3. Get Chat History
GET /chat
Description: Retrieves chat history for a user session.
Error Handling
The application includes error handling for various scenarios, including:
Invalid database URIs.
Connection failures.
Query execution errors.
