# Database Interaction and Chat Application

This application is built using **FastAPI** and enables users to connect to both SQL and NoSQL databases, perform queries, and interact through a chat interface. It leverages **Langchain** components for database interactions and response generation.

---

## **Project Setup and Installation**

### **Pre-requisites**

Before starting, ensure the following are in place:

- **Python Version:** 3.10 or higher
- **Docker:** Make sure Docker is installed and running on your server.
- **Docker Compose:** Ensure Docker Compose is installed for building and running the containers.

---

## **Installation and Setup**

### **Step 1: Clone the Repository**

Start by cloning this repository to your server using the following command:

```bash
git clone https://github.com/MuhammadAdil-Developer/langflow-chatbot
cd langflow-chat

Step 2: Build the Project with Docker Compose
Once you have cloned the repository, navigate to the project directory and run the following command to build the project using Docker Compose:

bash
Copy code
docker-compose up --build
This command will:

Build the Docker containers specified in the docker-compose.yml file.
Install all necessary dependencies as defined in the Dockerfile.
Step 3: Verify the Build
After running the build, ensure there are no dependency errors or issues. If the build completes successfully, Docker Compose will automatically start the application in the containers.

If there are any errors during the build,so resolve them before proceeding.

Running the Application
Once the project is built , follow these steps to run the FastAPI application:


The server should already be running if you followed the previous step. However, if you need to manually start it, you can use the following command inside the container:

bash
Copy code
uvicorn main:app --reload
Access the Application

After the server starts, you can access the application through your server’s IP address. Open a web browser and go to:

arduino
Copy code
http://<your_server_ip>:8080
Replace <your_server_ip> with your server's actual IP address.

You can also access the interactive API documentation at:

arduino
Copy code
http://<your_server_ip>:8080/docs
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
The application includes error handling for various scenarios, including:

Invalid database URIs.
Connection failures.
Query execution errors.
Future Enhancements
Add support for additional database types.
Improve chat interface with more conversational AI features.
Implement authentication for secure API access.
Troubleshooting
If you encounter any issues during setup or while running the project, check the following:

Docker Compose Build Errors:

Ensure Docker is properly installed and running.
Check for any missing dependencies or errors in the Dockerfile or docker-compose.yml.