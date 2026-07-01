import os
import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv

# Load env variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "deepshield")

db_config = {
    "host": DB_HOST,
    "port": DB_PORT,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME,
    "charset": "utf8mb4",
    "use_pure": True
}

# Create connection pool
# Note: Database 'deepshield' must exist before using connection pool with it.
# We will create a fallback connection pool logic without the database name to allow migrate.py to create the DB first.
db_pool = None

def get_pool():
    global db_pool
    if db_pool is None:
        try:
            db_pool = pooling.MySQLConnectionPool(
                pool_name="deepshield_pool",
                pool_size=10,
                pool_reset_session=True,
                **db_config
            )
        except mysql.connector.Error as err:
            # If the database does not exist yet (e.g., first run before migration),
            # initialize pool without database name so we can create it
            if err.errno == 1049:  # Unknown database error
                temp_config = db_config.copy()
                temp_config.pop("database", None)
                db_pool = pooling.MySQLConnectionPool(
                    pool_name="deepshield_pool_init",
                    pool_size=5,
                    pool_reset_session=True,
                    **temp_config
                )
            else:
                raise err
    return db_pool

def get_db_connection():
    pool = get_pool()
    return pool.get_connection()

def execute_query(query, params=None, commit=False, fetch="all"):
    """
    Executes a query safely using a connection from the pool.
    Parameters are parameterized to prevent SQL Injection.
    Returns:
        - List of dictionaries for fetch='all'
        - Dictionary (or None) for fetch='one'
        - Rowcount/LastInsertId for commit=True (as a dict)
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        
        if commit:
            conn.commit()
            result = {
                "lastrowid": cursor.lastrowid,
                "rowcount": cursor.rowcount
            }
        else:
            if fetch == "one":
                result = cursor.fetchone()
            elif fetch == "all":
                result = cursor.fetchall()
            else:
                result = None
                
        return result
    except mysql.connector.Error as err:
        if conn and commit:
            conn.rollback()
        print(f"[DATABASE ERROR] {err}")
        raise err
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()  # Returns connection back to the pool
