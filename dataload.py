import numpy as np
import psycopg2
import io
import utils

def load_dataset(TABLE_NAME,file,limit:int,conn):
    cur=conn.cursor()
    cur.execute(f"""
        DROP TABLE IF EXISTS {TABLE_NAME};
        CREATE TABLE {TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            embedding vector(128)
        );
    """)
    conn.commit()
    print(f"Table '{TABLE_NAME}' created.")

    print(f"Loading SIFT1M from {file} ...")
    vectors = utils.read_fvecs(file,limit)
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
