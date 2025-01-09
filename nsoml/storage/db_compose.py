import os
import sys
import psycopg2 # type: ignore

# Establish a connection to the PostgreSQL database
# Load connection details from environment variables
host = os.environ.get("DB_HOST")
port = os.environ.get("DB_PORT")
database = os.environ.get("DB_DATABASE")
user = os.environ.get("DB_USER")
password = os.environ.get("DB_PASSWORD")

# Establish a connection to the PostgreSQL database
conn = psycopg2.connect(
    # host="127.0.0.1",
    # port=5432,
    # database="chml",
    # user="postgres",
    # password="password"
    host=host,
    port=port,
    database=database,
    user=user,
    password=password,
)

# Create a cursor object
cur = conn.cursor()

if len(sys.argv) >= 2:
    if sys.argv[1] == "--reset":
        cur.execute("DROP TABLE IF EXISTS model_data, record, analysis, record_data;")
        conn.commit()
    elif sys.argv[1]:
        print("Invalid argument. Use '--reset' to reset the database.")
        sys.exit()

# Read the SQL script
with open("./schema.sql", 'r') as file:
    sql_script = file.read()
    print(sql_script)

# Execute the SQL script
cur.execute(sql_script)

# Commit the changes
conn.commit()

# Close the cursor and connection
cur.close()
conn.close()
