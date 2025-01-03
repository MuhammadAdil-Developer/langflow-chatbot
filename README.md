# Database Interaction and Chat Application

This application is built using **FastAPI** and enables users to connect to both SQL and NoSQL databases, perform queries, and interact through a chat interface. It leverages **Langchain** components for database interactions and response generation.

---

## **Requirements**

### **System Requirements**
- **Python Version:** Python 3.10 or higher
- **Required Python Packages:** See the list below.

---

## **Required Packages**

To install the necessary packages, follow these steps:

1. Create a `requirements.txt` file with the following content:
    ```text
    fastapi
    uvicorn
    mysql-connector-python
    pymongo
    langchain
    langflow
    ```

2. Install the packages using:
    ```bash
    pip install -r requirements.txt
    ```

---

## **Application Structure**

The application consists of the following components:

1. **FastAPI App**
   - The main application responsible for handling incoming requests and managing API endpoints.

2. **Database Connection**
   - Functions for connecting to:
     - **SQL Databases:** (e.g., MySQL)
     - **NoSQL Databases:** (e.g., MongoDB)

3. **Agent Components**
   - Components to create SQL and NoSQL agents for handling database queries and responses.

4. **User Session Management**
   - Functions for managing user sessions and chat histories.

---

## **Running the Application**

### **Step 1: Setting Up Environment Variables**

Before running the application, set up your environment variables for database connections and API keys. Create a `.env` file in the root directory with the following content:

```text
# Example .env file content
SQL_DATABASE_URI="your_sql_database_uri"
NOSQL_DATABASE_URI="your_nosql_database_uri"
API_KEY="your_api_key"
Step 2: Starting the Server
Run the FastAPI application using the following command:

bash
Copy code
uvicorn main:app --reload
Step 3: Accessing the API
Access the API at: http://127.0.0.1:8000
Interactive API documentation is available at: http://127.0.0.1:8000/docs
API Endpoints
1. Connect to Database
Endpoint: POST /connect-db
Description: Connects to a specified SQL or NoSQL database.
Request Body:
json
Copy code
{
  "question": "Your question here"
}
2. Chat Interface
Endpoint: POST /chat
Description: Sends a question to the connected database and retrieves a response.
Request Body:
json
Copy code
{
  "question": "Your question here",
  "db_name": "Database name",
  "database_uri": "Database URI",
  "collection_name": "Collection name",
  "db_type": "sql or nosql"
}
3. Get Chat History
Endpoint: GET /chat
Description: Retrieves the chat history for a user session.
Error Handling
The application includes error handling for various scenarios, such as:

Invalid database URIs.
Connection failures.
Query execution errors.
Future Enhancements
Add support for additional database types.
Improve chat interface with more conversational AI features.
Implement authentication for secure API access.
