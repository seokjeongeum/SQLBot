import os
import re
import sqlite3
import psycopg2
import argparse
from psycopg2.extras import execute_batch
from psycopg2 import sql

# Set up the argument parser
parser = argparse.ArgumentParser(description="Migrate SQLite databases to PostgreSQL")
parser.add_argument("search_dir", help="Directory where SQLite files are located")
args = parser.parse_args()

# PostgreSQL connection details
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5434"
POSTGRES_USER = "sqlbot"
POSTGRES_PASSWORD = "sqlbot_pw"


def create_postgres_db_name(sqlite_file_path):
    """
    Create a PostgreSQL database name from a SQLite file path
    """
    base = os.path.basename(sqlite_file_path)
    db_name = os.path.splitext(base)[0]
    db_name = re.sub(r"[^A-Za-z0-9_]", "_", db_name)
    db_name = f"{db_name}"
    return db_name


def sqlite_to_postgres(sqlite_db_path, postgres_db_name):
    """
    Migrate SQLite DB to PostgreSQL without including foreign key and primary key information.
    """
    # Connect to the SQLite database
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    sqlite_cursor = sqlite_conn.cursor()

    # Connect to the PostgreSQL database
    postgres_conn = psycopg2.connect(
        dbname=postgres_db_name,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
    )
    postgres_conn.autocommit = True
    postgres_cursor = postgres_conn.cursor()

    # Retrieve a list of tables from SQLite
    sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = sqlite_cursor.fetchall()

    for (table_name,) in tables:
        # Retrieve the SQL schema for each SQLite table
        sqlite_cursor.execute(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        )
        create_table_sql = sqlite_cursor.fetchone()[0]

        # Modify the SQL schema for compatibility with PostgreSQL
        create_table_sql = create_table_sql.replace("AUTOINCREMENT", "SERIAL")

        # Remove primary key constraints
        create_table_sql = re.sub(
            r",?\s*PRIMARY KEY\s*\([^)]+\)", "", create_table_sql, flags=re.IGNORECASE
        )

        # Remove foreign key constraints
        create_table_sql = re.sub(
            r",?\s*FOREIGN KEY\s*\([^)]+\)\s*REFERENCES\s*\S+\s*\([^)]+\)",
            "",
            create_table_sql,
            flags=re.IGNORECASE,
        )

        # Remove quotes and schema prefix
        create_table_sql = re.sub(r"\"[^\"]+\"\.", "", create_table_sql)

        # Execute the modified schema in PostgreSQL
        try:
            postgres_cursor.execute(create_table_sql.lower())
        except Exception as e:
            print(f"Error executing table schema: {e}")
            print(f"SQL: {create_table_sql}")
            continue

        # Retrieve data from SQLite table
        sqlite_cursor.execute(f"SELECT * FROM {table_name.lower()};")
        rows = sqlite_cursor.fetchall()
        columns = [desc[0].lower() for desc in sqlite_cursor.description]

        # Generate the SQL for inserting data
        placeholders = ", ".join(["%s" for _ in columns])
        insert_sql = f"INSERT INTO {table_name.lower()} ({', '.join(columns)}) VALUES ({placeholders});"

        try:
            # Attempt to insert data into PostgreSQL table
            execute_batch(postgres_cursor, insert_sql, rows)
        except Exception as e:
            print(f"Error inserting data: {e}")
            continue

    # Close connections
    sqlite_conn.close()
    postgres_conn.close()


def create_database(host, port, user, password, db_name):
    # Connect to the default database
    conn = psycopg2.connect(
        host=host, port=port, user=user, password=password, dbname="postgres"
    )
    conn.autocommit = (
        True  # This line is crucial, it sets the connection to autocommit mode
    )
    cur = conn.cursor()

    # Check if the database already exists
    cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,))
    exists = cur.fetchone()
    if not exists:
        # This line avoids the creation of a database if it already exists
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))

    cur.close()
    conn.close()


def main(search_dir):
    # Create the target PostgreSQL database
    if not os.path.isdir(search_dir):
        print(f"The directory {search_dir} does not exist.")
        exit(1)

    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith(".sqlite") or file.endswith(".db"):
                sqlite_file_path = os.path.join(root, file)
                print(f"Migrating {sqlite_file_path}...")
                postgres_db_name = create_postgres_db_name(sqlite_file_path)
                create_database(
                    POSTGRES_HOST,
                    POSTGRES_PORT,
                    POSTGRES_USER,
                    POSTGRES_PASSWORD,
                    postgres_db_name,
                )
                sqlite_to_postgres(sqlite_file_path, postgres_db_name)

    print(f"Migration complete for all SQLite files in {search_dir}.")


if __name__ == "__main__":
    main(args.search_dir)
