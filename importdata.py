# import pyodbc

# # SQL Server connection setup
# conn = pyodbc.connect(
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=localhost;"
#     "DATABASE=MyDatabase;"
#     "UID=SA;"
#     "PWD=admin@123"
# )
# cursor = conn.cursor()

# # Pehle Files table create karein
# try:
#     cursor.execute("""
#         CREATE TABLE Files (
#             ID INT IDENTITY(1,1) PRIMARY KEY,
#             FileName NVARCHAR(255),
#             FileContent VARBINARY(MAX)
#         )
#     """)
#     conn.commit()
#     print("Table created successfully!")
# except Exception as e:
#     print(f"Table creation error (ignore if table already exists): {str(e)}")

# # .bak file ko read karein aur store karein
# file_path = 'DEMODB2.bak'
# with open(file_path, 'rb') as file:
#     file_content = file.read()

# file_name = 'DEMODB2.bak'
# cursor.execute(
#     "INSERT INTO Files (FileName, FileContent) VALUES (?, ?)",
#     (file_name, file_content)
# )
# conn.commit()
# print("File successfully store kar di gayi!")
# conn.close()



import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=MyDatabase;UID=SA;PWD=admin@123"
)
cursor = conn.cursor()

# Query to fetch binary data
cursor.execute("SELECT FileContent FROM Files WHERE FileName = 'DEMODB2.bak'")
row = cursor.fetchone()

if row:
    with open("/var/opt/mssql/backups/DEMODB2.bak", "wb") as f:
        f.write(row[0])

print("File exported successfully!")
