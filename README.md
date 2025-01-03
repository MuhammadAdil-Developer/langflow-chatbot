# Database Interaction and Chat Application

This application is built using **FastAPI** and enables users to connect to both SQL and NoSQL databases, perform queries, and interact through a chat interface. It leverages **LangFlow** components for database interactions and response generation.

---

## **Project Setup and Installation**

### **Pre-requisites**
Before starting, ensure the following are in place:
- **Python Version:** 3.10 or higher
- **Docker:** Make sure Docker is installed and running on your server machine
- **Docker Compose:** Ensure Docker Compose is installed for building and running the containers

---

## **Installation and Setup**

### **Step 1: Clone the Repository**
Start by cloning this repository to your server using the following command:

```bash
git clone https://github.com/MuhammadAdil-Developer/langflow-chatbot
cd langflow-chat
```

### **Step 2: Run with Docker Compose**

#### **Option 1: Run in Attached Mode (with logs)**
```bash
docker-compose up --build
```
This command will:
- Build and start container
- Show real-time logs in your terminal
- Useful for initial testing and debugging

#### **Option 2: Run in Detached Mode (background)**
```bash
docker-compose up -d --build
```
This will:
- Build and start container in the background
- Allow continued use of your terminal
- Containers run in the background

### **Step 3: Managing Your Containers**

**View container status:**
```bash
docker-compose ps
```

**Check logs in detached mode:**
```bash
docker-compose logs -f
```

**Stop containers:**
```bash
docker-compose stop
```

**Remove containers (preserves data):**
```bash
docker-compose down
```

**Restart services:**
```bash
docker-compose restart
```

---

## **Accessing the Application**

After the server starts, you can access the application through:

- **Main Application:** `http://your-seerver-ip:8080/chat`
- **API Documentation:** `http://your-seerver-ip/docs`

---

## **API Endpoints**

### **1. Connect to Database**
- **Endpoint:** `POST /connect-db`
- **Description:** Connects to a specified SQL or NoSQL database
- **Request Body:**
```json
{
  "question": "Your question here"
}
```

### **2. Chat Interface**
- **Endpoint:** `POST /chat`
- **Description:** Sends a question to the connected database and retrieves a response
- **Request Body:**
```json
{
  "question": "Your question here",
  "db_name": "Database name",
  "database_uri": "Database URI",
  "collection_name": "Collection name",
  "db_type": "sql or nosql"
}
```

### **3. Get Chat History**
- **Endpoint:** `GET /chat`
- **Description:** Retrieves the chat history for a user session

---

## **Error Handling**

The application includes error handling for various scenarios, including:
- Invalid database URIs
- Connection failures
- Query execution errors

---

## **Future Enhancements**
- Add support for additional database types
- Improve chat interface with more conversational AI features
- Implement authentication for secure API access

---

## **Troubleshooting**

If you encounter any issues during setup or while running the project, check the following:

### **Docker Compose Build Errors:**
- Ensure Docker is properly installed and running
- Check for any missing dependencies
- Verify network connectivity for database connections
- Ensure required ports are available
