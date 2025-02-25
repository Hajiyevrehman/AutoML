# pipeline/data_loader.py

import os
import pandas as pd
import sqlite3

def get_absolute_path(file_path: str) -> str:
    """
    Returns an absolute path for a given file_path.
    """
    if not os.path.isabs(file_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(project_root, file_path)
    return file_path

def load_csv(file_path: str) -> pd.DataFrame:
    file_path = get_absolute_path(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {e}")

def load_sqlite(file_path: str, table_name: str = None) -> pd.DataFrame:
    file_path = get_absolute_path(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SQLite database file not found: {file_path}")
    conn = sqlite3.connect(file_path)
    try:
        if table_name is None:
            query_tables = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql_query(query_tables, conn)['name'].tolist()
            if not tables:
                raise ValueError("No tables found in the SQLite database.")
            elif len(tables) == 1:
                table_name = tables[0]
            else:
                raise ValueError(f"Multiple tables found: {tables}. Specify a table name.")
        query = f'SELECT * FROM "{table_name}"'
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        raise ValueError(f"Error reading table '{table_name}': {e}")
    finally:
        conn.close()
    return df

def load_data(file_path: str, file_type: str = None, table_name: str = None) -> pd.DataFrame:
    """
    Loads data by automatically detecting file type (csv or sqlite)
    if file_type is not provided. Otherwise, uses the specified file_type.
    """
    if file_type is not None:
        file_type = file_type.lower()
        if file_type == 'csv':
            return load_csv(file_path)
        elif file_type == 'sqlite':
            return load_sqlite(file_path, table_name)
        else:
            raise ValueError("Unsupported file_type. Use 'csv' or 'sqlite'.")
    else:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return load_csv(file_path)
        elif ext in ['.sqlite', '.db']:
            return load_sqlite(file_path, table_name)
        else:
            # Try CSV first, then fallback to SQLite
            try:
                return load_csv(file_path)
            except Exception:
                return load_sqlite(file_path, table_name)


