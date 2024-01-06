import sqlite3
from sqlite3 import Error

def create_connection():
    """
    Creates and returns a connection to the SQLite database.
    Returns:
        Connection: A connection object to the SQLite database, or None if a connection cannot be established.
    """
    try:
        conn = sqlite3.connect('features.db')
        return conn
    except Error as e:
        print(e)
        return None

def save_embedding(id, embedding_vector):
    """
    Saves or updates an embedding vector in the database for a given ID.
    Parameters:
        id (str): The unique identifier for the subject.
        embedding_vector (str): The embedding vector to be saved.
    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    # Create a connection to the database
    conn = create_connection()
    if conn is not None:
        try:
            # SQL command to insert or replace the embedding vector
            sql = ''' INSERT OR REPLACE INTO embeddings(id, embedding_vector) VALUES(?,?) '''
            cur = conn.cursor()
            cur.execute(sql, (id, embedding_vector))
            conn.commit()
            return True
        except Error as e:
            print(e)
            return False
        finally:
            # Ensure that the connection is closed even if an error occurs
            conn.close()
    else:
        return False

# Placeholder functions for additional CRUD operations
# def update_embedding(id, embedding_vector):
# def delete_embedding(id):
# def get_embedding(id):
