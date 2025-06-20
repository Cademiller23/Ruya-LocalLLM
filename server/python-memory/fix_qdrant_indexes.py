from qdrant_client import QdrantClient
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Qdrant client
if QDRANT_API_KEY:
    qdrant_client = QdrantClient(
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
    )
else:
    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

def fix_collection_indexes():
    """Add missing indexes to existing collections"""
    
    collections_to_fix = ["ruya_memories", "ruya_threads", "ruya_summaries"]
    
    for collection_name in collections_to_fix:
        try:
            # Check if collection exists
            collections = qdrant_client.get_collections().collections
            if not any(c.name == collection_name for c in collections):
                logger.info(f"Collection {collection_name} doesn't exist, skipping...")
                continue
            
            logger.info(f"Fixing indexes for collection: {collection_name}")
            
            # Create indexes for filtering
            index_fields = ["user_id", "workspace_id"]
            
            for field in index_fields:
                try:
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema="keyword"
                    )
                    logger.info(f"✅ Created index for {field} in {collection_name}")
                except Exception as e:
                    logger.info(f"ℹ️ Index for {field} might already exist: {str(e)}")
            
            # Add created_at index for memories collection
            if collection_name == "ruya_memories":
                try:
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="created_at",
                        field_schema="datetime"
                    )
                    logger.info(f"✅ Created datetime index for created_at in {collection_name}")
                except Exception as e:
                    logger.info(f"ℹ️ Datetime index might already exist: {str(e)}")
                    
        except Exception as e:
            logger.error(f"❌ Error fixing collection {collection_name}: {str(e)}")

if __name__ == "__main__":
    print("=" * 50)
    print("🔧 QDRANT INDEX FIX SCRIPT")
    print("=" * 50)
    logger.info("Starting Qdrant index fix...")
    fix_collection_indexes()
    logger.info("✅ Qdrant index fix completed!")
    print("=" * 50)