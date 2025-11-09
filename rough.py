import numpy as np
import psycopg2
import time
import utils

DB_NAME = "testdb"
DB_USER = "user1"
DB_PASS = "password1"
DB_HOST = "72.61.170.113"
DB_PORT = "5432"
TABLE_NAME = "sift1m"
SIFT_FILE = "sift/sift_base.fvecs"

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

print(f"Loading query vectors from {SIFT_FILE}...")
query_vecs = utils.read_fvecs(SIFT_FILE)

top_k_vecs = 5
all_query_results = []

print(f"Running exact k-NN (k=5) queries... total queries :",len(query_vecs))
start_time = time.time()
count = 0
for vec in query_vecs:
    count+=1
    query_vector_string = str(vec.tolist()) 
    cur.execute(f"""
        SELECT id FROM {TABLE_NAME}
        ORDER BY embedding <-> %s
        LIMIT {top_k_vecs};
    """, (query_vector_string,)) 
    if (count % 1000) == 0:
        print("processed queries :",count)
    # results = cur.fetchall()
    # all_query_results.append(results)

end_time = time.time()
print(f"Time taken for {len(query_vecs)} queries: {end_time - start_time:.4f} seconds")

# # Optional: Print the results
# print("\nResults (Nearest neighbor IDs):")
# for i, results in enumerate(all_query_results):
#     print(f"Query {i+1}: {results}")

cur.close()
conn.close()