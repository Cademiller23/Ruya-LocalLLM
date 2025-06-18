"""
Main Memory server for RuyaAI
Provides memory storage and hybrid search capabilities
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import cohere
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import sqlite3
import numpy as np
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from config import (
    COHERE_API_KEY,
    MEM0AI_API_KEY,
    PORT,
    HOST,
    MEMORY_THRESHOLD,
    MAX_MEMORIES,
    DEBUG,
    LOG_LEVEL
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(debug=DEBUG)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('memories.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            workspace_id TEXT NOT NULL,
            thread_id TEXT,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


init_db()
# Pydantic models for request/response validation
class MemoryContextRequest(BaseModel):
    userId: str
    message: str
    workspaceId: str

class MemoryContextResponse(BaseModel):
    context: str
    memories_used: int

class MemoryRequest(BaseModel):
    userId: str
    message: str
    response: str
    workspaceId: str
    threadId: Optional[str] = None

class MemorySearchRequest(BaseModel):
    userId: str
    query: str
    searchType: str = "hybrid"
    limit: int = MAX_MEMORIES
    workspaceId: str

class MemoryAddRequest(BaseModel):
    userId: str
    content: str
    metadata: Dict[str, Any] = {}
    workspaceId: str

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/context", response_model=MemoryContextResponse)
async def get_chat_context(request: MemoryContextRequest):
    try:
        # 1. Special case: Directly "What is my name?" query
        if "what is my name" in request.message.lower():
            conn = sqlite3.connect('memories.db')
            c = conn.cursor()
            c.execute('''
            SELECT metadata FROM memories
            WHERE user_id = ? AND workspace_id = ?
            ORDER BY created_at DESC
            ''', (request.userId, request.workspaceId))
            memories = c.fetchall()
            conn.close()

            for mem in memories:
                meta = json.loads(mem[0]) if mem[0] else {}
                name = meta.get('username')
                if name and name.lower() != "my name is":
                    return {
                        "context": f"User's name is {name}.",
                        "memories_used": 1
                    }
        # Fallback
        # Get message embedding
        response = co.embed(texts=[request.message], model='embed-english-v3.0', input_type='search_document')
        message_embedding = response.embeddings[0]
        if len(message_embedding) != 1024:
            raise ValueError("Embedding size mismatch: expected 1024, got {}".format(len(message_embedding)), status_code=500)
        # Search for relevant memories
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT content, metadata, embedding
            FROM memories
            WHERE user_id = ? AND workspace_id = ?
        ''', (request.userId, request.workspaceId))
        memories = c.fetchall()
        
        relevant_memories = []
        for memory, content, stored_embedding in memories:
            stored_embedding = np.frombuffer(stored_embedding, dtype=np.float32)
            if stored_embedding.shape[0] != 1024:
                logger.warning("Skipping memory with invalid embedding size: %s", content)
                continue
            similarity = np.dot(message_embedding, stored_embedding) / (
                np.linalg.norm(message_embedding) * np.linalg.norm(stored_embedding)
            )
            if similarity > MEMORY_THRESHOLD:
                relevant_memories.append({
                    'content': content,
                    'metadata': json.loads(metadata) if metadata else {}
                })

        conn.close()

        # Format context from relevant memories
        context_parts = [
            f"Memory {i + 1}: {m['content']} (Metadata: {json.dumps(m['metadata'])})"
            for i, m in enumerate(relevant_memories)

        ]
        return {
            'context': '\n'.join(context_parts),
            'memories_used': len(relevant_memories)
        }
    except Exception as e:
        logger.error("Error in get_chat_context: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory")
async def store_memory(request: MemoryRequest):
    def extract_username(text):
        import re
        match = re.search(r"\bmy name is ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b", text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            print(f"[Username Extracted] => {name}")
            return name
        print("[Username Extracted] => None")
        return None
    try:
        # Create memory content
        memory_content = "User: {}\nAssistant: {}".format(request.message, request.response)
        username = extract_username(request.message)
        metadata = {'type': 'chat'}
        if username:
            metadata['username'] = username
        # Get embedding
        response = co.embed(texts=[memory_content], model='embed-english-v3.0', input_type='search_document')
        embedding = response.embeddings[0]
        if len(embedding) != 1024:
            raise ValueError("Embedding size mismatch: expected 1024, got {}".format(len(embedding)), status_code=500)
        # Store in database
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO memories (user_id, content, metadata, workspace_id, thread_id, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            request.userId,
            memory_content,
            json.dumps(metadata),
            request.workspaceId,
            request.threadId,
            np.array(embedding).astype(np.float32).tobytes()
        ))
        
        memory_id = c.lastrowid
        conn.commit()
        conn.close()

        logger.info("Memory stored successfully with ID: %s", memory_id)
        return {"success": True, "id": memory_id}
    except Exception as e:
        logger.error("Error in store_memory: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_memories(request: MemorySearchRequest):
    try:
        # Get query embedding
        response = co.embed(texts=[request.query], model='embed-english-v3.0', input_type='search_query')
        query_embedding = response.embeddings[0]

        # Search memories
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        query = '''
            SELECT content, metadata, embedding
            FROM memories
            WHERE user_id = ? AND workspace_id = ?
        '''
        c.execute(query, (request.userId, request.workspaceId))
        memories = c.fetchall()
        
        results = []
        for memory in memories:
            content, metadata, stored_embedding = memory
            stored_embedding = np.frombuffer(stored_embedding, dtype=np.float32)
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            results.append({
                'content': content,
                'metadata': json.loads(metadata) if metadata else {},
                'similarity': float(similarity)
            })

        # Sort by similarity and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:request.limit]

        conn.close()
        return results
    except Exception as e:
        logger.error("Error in search_memories: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories")
async def get_memories(userId: str, workspaceId: str):
    try:
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        query = '''
            SELECT content, metadata, created_at
            FROM memories
            WHERE user_id = ? AND workspace_id = ?
            ORDER BY created_at DESC
        '''
        c.execute(query, (userId, workspaceId))
        memories = c.fetchall()
        
        results = [{
            'content': memory[0],
            'metadata': json.loads(memory[1]) if memory[1] else {},
            'created_at': memory[2]
        } for memory in memories]

        conn.close()
        return results
    except Exception as e:
        logger.error("Error in get_memories: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/add")
async def add_memory(request: MemoryAddRequest):
    try:
        # Get embedding
        response = co.embed(texts=[request.content], model='embed-english-v3.0', input_type='search_document')
        embedding = response.embeddings[0]
        if len(embedding) != 1024:
            raise ValueError("Embedding size mismatch: expected 1024, got {}".format(len(embedding)), status_code=500)
        # Store in database
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO memories (user_id, content, metadata, workspace_id, embedding)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            request.userId,
            request.content,
            json.dumps(request.metadata),
            request.workspaceId,
            np.array(embedding).tobytes()
        ))
        
        memory_id = c.lastrowid
        conn.commit()
        conn.close()

        logger.info("Memory added successfully with ID: %s", memory_id)
        print(f"Added memory ID {memory_id} for userId={request.userId}, workspaceId={request.workspaceId}, content={request.content}")
        return {"success": True, "id": memory_id}
    except Exception as e:
        logger.error("Error in add_memory: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/username")
async def get_username(userId: str, workspaceId: str):
    try:
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()

        c.execute('''
                  SELECT metadata FROM memories
                  WHERE user_id = ? AND workspace_id = ?
                  ORDER BY created_at DESC
                  ''', (userId, workspaceId))
        memories = c.fetchall()
        conn.close()

        for mem in memories:
            meta = json.loads(mem[0]) if mem[0] else {}
            name = meta.get('username')
            if name and name.lower() != "my name is":
                return {"username": name}
        
        return {'username': None}
    except Exception as e:
        logger.error("Error in get_username: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == '__main__':
    import uvicorn
    logger.info("Starting memory server on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)



