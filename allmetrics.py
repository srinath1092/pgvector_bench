from math import log
import time
from venv import logger
import numpy as np
import utils
import psycopg2 

def dist_query(query_vecs:np.ndarray,top_k_vecs:int,dist_func,cur:psycopg2.extensions.cursor,TABLE_NAME,logger:utils.clogger):
    for vec in query_vecs:
        query_vector_string = str(vec.tolist()) 
        execute_for_time(f"""
            SELECT id FROM {TABLE_NAME}
            ORDER BY embedding {utils.dist_functions[dist_func]} %s
            LIMIT {top_k_vecs};""",cur,logger,args=(query_vector_string,))

def execute_for_time(query:str,cur:psycopg2.extensions.cursor,logger:utils.clogger,args=()):
    cur.execute(f"EXPLAIN ANALYZE {query}",args) 
    plan=cur.fetchall()
    for row in plan:
        if "Time" not in row[0]: continue
        logger.write(row[0])

def execute_for_time_2(query:str,cur:psycopg2.extensions.cursor,logger:utils.clogger,args=()):
    startTime=time.time()
    cur.execute(f"{query}",args) 
    cur.connection.commit()
    endTime=time.time()
    logger.write(endTime-startTime)

def dummy_index_build(cur:psycopg2.extensions.cursor,logger:utils.clogger):
    execute_for_time_2("CREATE INDEX ON sift1m USING hnsw (embedding vector_l2_ops);",cur,logger)
