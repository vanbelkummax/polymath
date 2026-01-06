#!/usr/bin/env python3
"""
Simple database wrapper for Polymath system.
Provides clean API for common operations.
"""
import psycopg2
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Database:
    """Simple database wrapper with connection pooling."""

    def __init__(self, dsn: str = "dbname=polymath user=polymath host=/var/run/postgresql"):
        self.dsn = dsn
        self._conn = None

    def _get_conn(self):
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
        return self._conn

    def execute(self, query: str, params: tuple = None) -> None:
        """Execute a query without returning results."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error executing query: {e}")
            raise
        finally:
            cursor.close()

    def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row as a dictionary."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            row = cursor.fetchone()
            if row is None:
                return None

            # Convert to dict using column names
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        finally:
            cursor.close()

    def fetch_all(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Fetch all rows as dictionaries."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to list of dicts
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cursor.close()

    def insert(self, query: str, params: tuple = None) -> int:
        """Execute INSERT and return the ID from RETURNING clause."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            conn.commit()
            return result[0] if result else None
        except Exception as e:
            conn.rollback()
            logger.error(f"Error inserting: {e}")
            raise
        finally:
            cursor.close()

    def query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Alias for fetch_all."""
        return self.fetch_all(query, params)

    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()


# Global instance
db = Database()


def get_db_connection():
    """Get a psycopg2 connection (for compatibility with older code)."""
    return db._get_conn()
