"""
Add HNSW index on DocumentChunk.embedding for fast approximate nearest neighbor search.

Without this index, every vector similarity query does a sequential scan across ALL chunks,
computing L2 distance on 768-dimensional vectors for each row. With HNSW, this becomes
an O(log n) approximate search that returns results in <100ms instead of 10-20 seconds.
"""

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0025_reduce_embedding_dimensions"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
                CREATE INDEX IF NOT EXISTS idx_chunk_embedding_hnsw
                ON documents_documentchunk
                USING hnsw (embedding vector_l2_ops)
                WITH (m = 16, ef_construction = 64);
            """,
            reverse_sql="DROP INDEX IF EXISTS idx_chunk_embedding_hnsw;",
        ),
    ]

