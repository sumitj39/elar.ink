-- Copy from a csv file to a table
COPY temp_embeddings(entry_id, embedding, content)
FROM '/Users/sumit.jogalekar/suprdaily/en2kan/embeddings-data-gte-base-all.csv' DELIMITER ',' CSV HEADER;


-- create an hnsw index on embeddings vector
CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops);

-- set maintenance_work_mem to 1GB
SET maintenance_work_mem TO '1GB';
