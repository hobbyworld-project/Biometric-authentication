import sqlite3

def init_db():
    """
    Initializes the database by creating a table for embeddings.
    This function creates a SQLite database and a table if they don't already exist.
    The table 'embeddings' includes two fields: 'id' and 'embedding_vector'.
    """
    # Connect to the SQLite database (creates the file if it doesn't exist)
    with sqlite3.connect('features.db') as connection:
        cursor = connection.cursor()

        # SQL command to create the 'embeddings' table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            embedding_vector TEXT NOT NULL
        )
        ''')

        # Commit the changes to the database
        connection.commit()

# The following block ensures that init_db() is called only when the script is executed directly
if __name__ == "__main__":
    init_db()
