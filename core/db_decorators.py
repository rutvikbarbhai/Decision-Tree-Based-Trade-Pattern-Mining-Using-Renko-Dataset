import traceback
import sqlalchemy as sql
import pandas as pd
import time
import random
from functools import wraps
import psycopg2
from psycopg2.errors import SerializationFailure
from .logs import logger


def with_db_transaction(func):  # noqa: C901
    @wraps(func)
    def wrapper(*args, **kwargs):
        query = kwargs.get('query')
        params = kwargs.get('params', {})
        engine = kwargs.get('engine')
        max_retries = kwargs.get('max_retries', 3)
        write_op = kwargs.get('write', False)
        if not engine:
            raise ValueError("An SQLAlchemy engine must be provided")
        for connection_retry in range(1, max_retries + 1):
            try:
                with engine.connect() as conn:
                    # Begin a transaction
                    transaction = conn.begin()
                    try:
                        # Call the decorated function with the connection
                        result = func(conn, *args, **kwargs)
                        if write_op:
                            # Commit any changes if the operation is a write operation
                            transaction.commit()
                        return result
                    except Exception as exc:
                        logger.error(f'Error while processing query {query} with params: {params}: {exc}')
                        logger.info(traceback.format_exc())
                        try:
                            transaction.rollback()  # Rollback if there's an error
                        except Exception as rollback_exc:
                            logger.error(f'Error during rollback: {rollback_exc}')
                        raise exc
            except Exception as exc:
                logger.error(f'Error while processing query {query}: {exc}')
                logger.info(traceback.format_exc())
                logger.info(f"DB Connection Retry: {connection_retry}")
                sleep_ms = random_sleep_time(connection_retry)
                logger.info(f"Sleeping {sleep_ms} seconds")
                time.sleep(sleep_ms)

        raise ValueError(f"DB Connection not succeed after {max_retries} retries")
    return wrapper


def with_raw_db_transaction_retries(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        query = kwargs.get('query')
        engine = kwargs.get('engine')
        params = kwargs.get('params', {})
        max_retries = kwargs.get('max_retries', 3)
        write_op = kwargs.get('write', False)
        for connection_retry in range(1, max_retries + 1):
            try:
                with engine.connect() as conn:
                    dbapi_conn = conn.connection
                    for retry in range(1, max_retries + 1):
                        try:
                            result = func(conn, *args, **kwargs)
                            if write_op:
                                dbapi_conn.commit()  # Commit any changes if there were any
                            return result
                        except SerializationFailure as exc:
                            # This is a retry error, so we roll back the current
                            # transaction and sleep for a bit before retrying. The
                            # sleep time increases for each failed transaction.
                            logger.error(f'Serialization error while processing query {query} with params: {params}: {exc}')
                            dbapi_conn.rollback()  # Rollback the transaction in case of error
                            logger.info("EXECUTE SERIALIZATION_FAILURE BRANCH")
                            logger.info(f"DB Query Retry: {retry}")
                            sleep_ms = random_sleep_time(retry)
                            logger.info(f"Sleeping {sleep_ms} seconds")
                            time.sleep(sleep_ms)
                        except psycopg2.Error as exc:
                            logger.error(f'Non-serialization error while processing query {query} with params: {params}: {exc}')
                            logger.info(traceback.format_exc())
                            raise exc
                    raise ValueError(f"DB query did not succeed after {max_retries} retries")
            except Exception as exc:
                logger.error(f'Error while processing query {query}: {exc}')
                logger.info(traceback.format_exc())
                logger.info(f"DB Connection Retry: {connection_retry}")
                sleep_ms = random_sleep_time(connection_retry)
                logger.info(f"Sleeping {sleep_ms} seconds")
                time.sleep(sleep_ms)

        raise ValueError(f"DB Connection not succeed after {max_retries} retries")
    return wrapper


def random_sleep_time(retry):
    '''Exponential backoff with jitter'''
    return (2**retry) * 0.1 * (random.random() + 0.5)


@with_raw_db_transaction_retries
def fetch_data_from_db(conn, query, params={}, engine=None):
    return pd.read_sql(query, params=params, con=conn)


@with_db_transaction
def fetch_one_from_db(conn, query, params={}, engine=None, write=False):
    return conn.execute(sql.text(query), params).mappings().fetchone()


@with_db_transaction
def fetch_all_from_db(conn, query, params={}, engine=None, write=False):
    return conn.execute(sql.text(query), params).mappings().fetchall()


@with_raw_db_transaction_retries
def fetch_data_from_cdb(conn, query, params={}, engine=None):
    return pd.read_sql(query, params=params, con=conn)


def with_db_transaction_cursor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        query = kwargs.get('query')
        engine = kwargs.get('engine')

        try:
            with engine.connect() as conn:
                dbapi_conn = conn.connection
                try:
                    with dbapi_conn.cursor() as cursor:
                        # Call the decorated function with the connection
                        result = func(cursor, *args, **kwargs)
                        dbapi_conn.commit()  # Commit any changes if there were any
                        return result
                except psycopg2.Error as exc:
                    logger.error(f'Error while processing query {query}: {exc}')
                    logger.info(traceback.format_exc())
                    if dbapi_conn:
                        try:
                            dbapi_conn.rollback()  # Rollback the transaction in case of error
                        except Exception as rollback_exc:
                            logger.error(f'Error during rollback: {rollback_exc}')
                            raise rollback_exc
        except Exception as exc:
            logger.error(f'Error while processing query {query}: {exc}')
            logger.info(traceback.format_exc())
    return wrapper


@with_db_transaction_cursor
def update_query_raw(cursor, query, engine=None, data=None):
    # Define the query and data
    logger.debug(f"executing {query} with {data}")
    cursor.execute(query, data)
    # Check how many rows were updated
    logger.info(f"Number of rows updated: {cursor.rowcount}")
    # Commit the transaction - using the raw db conn
    return cursor.rowcount
