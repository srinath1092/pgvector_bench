import numpy as np
import psycopg2
import io
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

cur.execute(f"""
    DROP TABLE IF EXISTS {TABLE_NAME};
    CREATE TABLE {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        embedding vector(128)
    );
""")
conn.commit()
print(f"Table '{TABLE_NAME}' created.")

print(f"Loading SIFT1M from {SIFT_FILE} ...")
vectors = utils.read_fvecs(SIFT_FILE)
print(f"Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}.")

print("Preparing data for COPY...")

f = io.StringIO()

for vec in vectors:
    f.write(f"[{','.join(map(str, vec))}]\n")

f.seek(0)

print("Executing COPY from STDIN...")
cur.copy_expert(f"COPY {TABLE_NAME} (embedding) FROM STDIN", f)

conn.commit()

print("Data loading successful.")

cur.close()
conn.close()