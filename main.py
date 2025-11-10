from pydoc import allmethods
import numpy as np
import psycopg2
import io
import allmetrics
import dataload
import utils

DB_NAME = "testdb"
DB_USER = "user1"
DB_PASS = "password1"
DB_HOST = "72.61.170.113"
DB_PORT = "5432"
TABLE_NAME = "sift1m"
SIFT_FILE = "/scratch/siftsmall/siftsmall_base.fvecs"

QUERY_FILE = "/scratch/siftsmall/siftsmall_base.fvecs"

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT
)
cur=conn.cursor()

dataload.load_dataset(TABLE_NAME,SIFT_FILE,10000,conn)


query_vectors = utils.read_fvecs(QUERY_FILE,100)

logger:utils.clogger=utils.clogger("log.log")

exact_search_results = {}

for topk in range(5,1000,20):
    for dist_func in utils.dist_functions.keys():
        logger.set_context(f"{dist_func}[{topk}][{len(query_vectors)}]")
        allmetrics.dist_query(query_vectors,topk,dist_func,cur,TABLE_NAME,logger)


# for max_cons in [4,8,16,32,64]:
#     for ef_construction_multiplier in [2,4,8]:
#         for dist_func in utils.dist_functions.keys():
#             logger.set_context(f"hnsw_{dist_func}_{max_cons}_{ef_construction_multiplier}")
#             allmetrics.build_hnsw_index(dist_func,max_cons,ef_construction_multiplier*max_cons,cur,logger)

