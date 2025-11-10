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

dataload.load_dataset(TABLE_NAME,SIFT_FILE,1000,conn)


query_vectors = utils.read_fvecs(QUERY_FILE,100)

logger:utils.clogger=utils.clogger("log.log")

exact_search_results = {}

for topk in range(80,101,20):
    for dist_func in utils.dist_functions.keys():
        logger.set_context(f"exact_search_{dist_func}[{topk}][{len(query_vectors)}]")
        allmetrics.dist_query(query_vectors,topk,dist_func,cur,TABLE_NAME,logger)
        exact_search_results[f"{topk}_{dist_func}"] = allmetrics.get_results(query_vectors,topk,dist_func,cur,TABLE_NAME,logger)

hnsw_search_results = {}
for ef_search in [20,40,60,80,100]:
    for max_cons in [4,8,16,32,64]:
        for ef_construction_multiplier in [2,4,8]:
            for dist_func in utils.dist_functions.keys():
                logger.set_context(f"hnsw_build_{dist_func}_{max_cons}_{ef_construction_multiplier*max_cons}")
                allmetrics.build_hnsw_index(dist_func,max_cons,ef_construction_multiplier*max_cons,cur,logger)
                logger.write(f"Memory {allmetrics.get_index_size("hnsw",cur)}")

                for topk in range(80,101,20):
                    logger.set_context(f"hnsw_query_{ef_search}_{dist_func}[{topk}][{len(query_vectors)}]")
                    allmetrics.dist_query_hnsw(query_vectors,topk,dist_func,cur,TABLE_NAME,logger,ef_search)
                    hnsw_search_results[f"{topk}_{dist_func}"] = allmetrics.get_results_hnsw(query_vectors,topk,dist_func,cur,TABLE_NAME,logger,ef_search)
            logger.set_context(f"hnsw_recall_{ef_search}_{max_cons}_{ef_construction_multiplier*max_cons}[{len(query_vectors)}]")
            utils.log_recall(exact_search_results,hnsw_search_results,logger)

ivf_search_results = {}
for n_probes in [1,5,10,15]:
    for ivf_list_count in [80,100,120]:
        for dist_func in utils.dist_functions.keys():
            if dist_func == "l1": continue
            logger.set_context(f"ivf_build_{dist_func}_{ivf_list_count}")
            allmetrics.build_ivf_index(dist_func,ivf_list_count,cur,logger)
            logger.write(f"Memory {allmetrics.get_index_size("ivf",cur)}")

            for topk in range(80,101,20):
                logger.set_context(f"ivf_query_{n_probes}_{dist_func}[{topk}][{len(query_vectors)}]")
                allmetrics.dist_query_ivf(query_vectors,topk,dist_func,cur,TABLE_NAME,logger,n_probes)
                ivf_search_results[f"{topk}_{dist_func}"] = allmetrics.get_results_ivf(query_vectors,topk,dist_func,cur,TABLE_NAME,logger,n_probes)
        logger.set_context(f"ivf_recall_{ivf_list_count}_{n_probes}[{len(query_vectors)}]")
        utils.log_recall(exact_search_results,ivf_search_results,logger)
