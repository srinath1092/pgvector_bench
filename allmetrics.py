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
        

def dist_query_ivf(query_vecs:np.ndarray,top_k_vecs:int,dist_func,cur:psycopg2.extensions.cursor,TABLE_NAME,logger:utils.clogger,n_probes):
    for vec in query_vecs:
        query_vector_string = str(vec.tolist()) 
        execute_for_time(f"""
            SET ivfflat.probes = {n_probes};
            SELECT id FROM {TABLE_NAME}
            ORDER BY embedding {utils.dist_functions[dist_func]} %s
            LIMIT {top_k_vecs};""",cur,logger,args=(query_vector_string,))

def get_results(query_vecs:np.ndarray,top_k_vecs:int,dist_func,cur:psycopg2.extensions.cursor,TABLE_NAME,logger:utils.clogger):
    results = []
    for vec in query_vecs:
        query_vector_string = str(vec.tolist()) 
        
        cur.execute(f"""
            SELECT id FROM {TABLE_NAME}
            ORDER BY embedding <-> %s
            LIMIT {top_k_vecs};
        """, (query_vector_string,))
        results.append(cur.fetchall())
    return results

def execute_for_time(query:str,cur:psycopg2.extensions.cursor,logger:utils.clogger,args=()):
    cur.execute(f"EXPLAIN (ANALYZE,BUFFERS) {query}",args) 
    print("executing ",query)
    plan=cur.fetchall()
    for row in plan:
        # print(row)
        if "Time" not in row[0]: continue
        logger.write(row[0])

    # exit(0)
def execute_for_time_2(query:str,cur:psycopg2.extensions.cursor,logger:utils.clogger,args=()):
    startTime=time.time()
    cur.execute(f"{query}",args) 
    cur.connection.commit()
    endTime=time.time()
    logger.write(f"Time {endTime-startTime}")


def build_index(index_name:str,dist_func:str,cur:psycopg2.extensions.cursor,logger:utils.clogger):
    cur.execute(f"DROP INDEX IF EXISTS {index_name}")
    cur.connection.commit()
    execute_for_time_2(f"CREATE INDEX {index_name} ON sift1m USING {index_name} (embedding vector_{dist_func}_ops);",cur,logger)

def build_hnsw_index(dist_func:str,max_cons:int,ef_construction:int,cur:psycopg2.extensions.cursor,logger:utils.clogger):
    print(logger._context,"started")
    cur.execute(f"DROP INDEX IF EXISTS hnsw")
    cur.execute(f"DROP INDEX IF EXISTS ivf")
    cur.connection.commit()
    execute_for_time_2(f"CREATE INDEX hnsw ON sift1m USING hnsw (embedding vector_{dist_func}_ops) with (m={max_cons},ef_construction={ef_construction});",cur,logger)
    print(logger._context,"done")

def build_ivf_index(dist_func:str,ivf_list_count:int,cur:psycopg2.extensions.cursor,logger:utils.clogger):
    print(logger._context,"started")
    cur.execute(f"DROP INDEX IF EXISTS hnsw")
    cur.execute(f"DROP INDEX IF EXISTS ivf")
    cur.connection.commit()
    execute_for_time_2(f"CREATE INDEX ivf ON sift1m USING ivfflat (embedding vector_{dist_func}_ops) with (lists={ivf_list_count});",cur,logger)
    print(logger._context,"done")



def get_index_size(index_name:str,cur:psycopg2.extensions.cursor):
    cur.execute(
    f"""SELECT
    indexrelname AS index_name,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
    FROM pg_stat_user_indexes
    WHERE indexrelname = '{index_name}';""")
    results=cur.fetchall()
    return results[0][1]
