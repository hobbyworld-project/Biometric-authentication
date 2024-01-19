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

def get_embedding(id):
    """
    Retrieves an embedding vector from the database for a given ID.
    Parameters:
        id (str): The unique identifier for the subject.
    Returns:
        embedding_vector (str): The embedding vector retrieved from the database, or None if not found or an error occurs.
    """
    # Create a connection to the database
    conn = create_connection()
    if conn is not None:
        try:
            # SQL command to select the embedding vector
            sql = ''' SELECT embedding_vector FROM embeddings WHERE id = ? '''
            cur = conn.cursor()
            cur.execute(sql, (id,))
            row = cur.fetchone()
            if row is not None:
                return row[0]  # Return the embedding_vector if found
            else:
                return None  # Return None if no record is found
        except Error as e:
            print(e)
            return None
        finally:
            # Ensure that the connection is closed even if an error occurs
            conn.close()
    else:
        return None

def get_all_embeddings():
    """
    Retrieves all embedding vectors from the database.
    Returns:
        embeddings (list of tuples): A list of (id, embedding_vector) tuples, or an empty list if not found or an error occurs.
    """
    # Create a connection to the database
    conn = create_connection()
    embeddings = []
    if conn is not None:
        try:
            # SQL command to select all id and embedding_vector pairs
            sql = ''' SELECT id, embedding_vector FROM embeddings '''
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            for row in rows:
                embeddings.append((row[0], row[1]))  # Append the (id, embedding_vector) tuple to the list
            return embeddings
        except Error as e:
            print(e)
            return embeddings  # Return the potentially partially filled list of embeddings
        finally:
            # Ensure that the connection is closed even if an error occurs
            conn.close()
    else:
        return embeddings