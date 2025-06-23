"""
Main Memory server for RuyaAI
Provides memory storage and hybrid search capabilities
Adding: Reranking, Narrative summaries, LT/ST memory, Structured Reasoning, Qdrant, Auto-traits
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import cohere
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import sqlite3
import numpy as np
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import logging
import re
from collections import defaultdict
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models
import hashlib
import uuid
from config import (
    COHERE_API_KEY,
    PORT,
    HOST,
    MEMORY_THRESHOLD,
    MAX_MEMORIES,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    RERANK_TOP_N,
    EMBEDDING_MODEL,
    RERANK_MODEL
)


# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI()

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

# Init enhanced Qdrant client and sqlite for memory storage
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
# Collection names
MEMORIES_COLLECTION = "ruya_memories"
THREADS_COLLECTION = "ruya_threads"
SUMMARIES_COLLECTION = "ruya_summaries"

# 

async def init_qdrant_collections():
    # Init qdrant collections if do not exist
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if MEMORIES_COLLECTION not in collection_names:
            qdrant_client.create_collection(
                collection_name=MEMORIES_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            

            qdrant_client.create_payload_index(
                collection_name=MEMORIES_COLLECTION,
                field_name="user_id",
                field_schema="keyword",

            )
            qdrant_client.create_payload_index(
                collection_name=MEMORIES_COLLECTION,
                field_name="workspace_id",
                field_schema="keyword",

            )
            qdrant_client.create_payload_index(
                collection_name=MEMORIES_COLLECTION,
                field_name="created_at",
                field_schema="datetime",

            )
            logger.info(f"Created Qdrant collection: {MEMORIES_COLLECTION}")
        else:
            # Check if indexes exist, if not create them
            try:
                collection_info = qdrant_client.get_collection(MEMORIES_COLLECTION)

                # Collection exists but indexes might be missing, create them
                qdrant_client.create_payload_index(
                    collection_name=MEMORIES_COLLECTION,
                    field_name="user_id",
                    field_schema="keyword"
                )
                qdrant_client.create_payload_index(
                    collection_name=MEMORIES_COLLECTION,
                    field_name="workspace_id",
                    field_schema="keyword"
                )
            except Exception as e:
                logger.info(f"Indexes might already exist: {str(e)}")
        
        # Similiar for others

        if THREADS_COLLECTION not in collection_names:
            qdrant_client.create_collection(
                collection_name=THREADS_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {THREADS_COLLECTION}")
        if SUMMARIES_COLLECTION not in collection_names:
            qdrant_client.create_collection(
                collection_name=SUMMARIES_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {SUMMARIES_COLLECTION}")
    except Exception as e:
        logger.error(f"Error initializing Qdrant collections: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize Qdrant collections")

# Init enhanced SQLITE Database
def init_enhanced_db():
    # Init enhanced database with additional fields
    conn = sqlite3.connect('memories.db')
    c = conn.cursor()

    # orig memories table
    c.execute('''
        CREATE TABLE IF NOT EXISTS memories (
              id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              content TEXT NOT NULL,
              metadata TEXT,
              memory_type TEXT DEFAULT 'short_term',
              workspace_id TEXT NOT NULL,
              thread_id TEXT,
              reasoning_trace TEXT,
              embedding BLOB,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              access_count INTEGER DEFAULT 0
              )
              ''')
    # User traits table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_traits (
              user_id TEXT PRIMARY KEY,
              name TEXT,
              location TEXT,
              timezone TEXT,
              role TEXT,
              preferences TEXT,
              goals TEXT,
              sentiment_profile TEXT,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
              )
             ''' )
    # Thread persistence table
    c.execute('''
        CREATE TABLE IF NOT EXISTS thread_states (
            thread_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            title TEXT,
            summary TEXT,
            messages TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
              ''')
    # Narrative summaries table
    c.execute('''
        CREATE TABLE IF NOT EXISTS narrative_summaries (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            linked_memories TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
              ''')
    # Create indexes for performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_memories_user_workspace ON memories(user_id, workspace_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_thread_states_thread ON thread_states(thread_id)')

    conn.commit()
    conn.close()

# Initialize database
init_enhanced_db()


# Pydantic models for request/response validation
class MemoryRequest(BaseModel):
    userId: str
    message: str
    response: str
    workspaceId: str
    threadId: Optional[str] = None
    reasoning: Optional[str] = None

class MemoryContextRequest(BaseModel):
    userId: str
    message: str
    workspaceId: str

class ThreadStateRequest(BaseModel):
    threadId: str
    workspaceId: str
    userId: str 
    title: Optional[str] = None
    messages: List[Dict[str, Any]]


# class MemoryContextResponse(BaseModel):
#     context: str
#     memories_used: int



# class MemorySearchRequest(BaseModel):
#     userId: str
#     query: str
#     searchType: str = "hybrid"
#     limit: int = MAX_MEMORIES
#     workspaceId: str

# class MemoryAddRequest(BaseModel):
#     userId: str
#     content: str
#     metadata: Dict[str, Any] = {}
#     workspaceId: str


# Utility classes
class UserTraits:
    # Auto extract and manage user traits

    @staticmethod 
    def extract_name(text: str) -> Optional[str]:
        patterns = [
            r"my name is ([A-Z][a-z]+(?: [A-Z][a-z]+)*)",
            r"i'm ([A-Z][a-z]+(?: [A-Z][a-z]+)*)",
            r"i am ([A-Z][a-z]+(?: [A-Z][a-z]+)*)",
            r"call me ([A-Z][a-z]+(?: [A-Z][a-z]+)*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    @staticmethod
    def extract_location(text: str) -> Optional[str]:
        patterns = [
            r"i (?:live|am) (?:in|at|from) ([A-Za-z\s,]+)",
            r"i'm (?:based|located) (?:in|at) ([A-Za-z\s,]+)",
            r"from ([A-Za-z\s,]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    @staticmethod
    def extract_role(text: str) -> Optional[str]:
        patterns = [
            r"i (?:am|work as) (?:a|an) ([A-Za-z\s]+)",
            r"my (?:job|role|position) is ([A-Za-z\s]+)",
            r"i'm (?:a|an) ([A-Za-z\s]+) (?:at|for|with)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    @staticmethod 
    async def analyze_sentiment(text: str) -> str:
        # Analyze sentiment using simple heuristics 
        positive_words = ['happy', 'great', 'love', 'excellent', 'wonderful', 'amazing']
        negative_words = ['sad', 'angry', 'hate', 'terrible', 'awful', 'horrible']

        text_lower = text.lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)

        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
class MemoryManager:
    # Memory amangeement with ST/LT memory

    @staticmethod 
    def determine_memory_type(content: str, access_count: int, created_at: str) -> str:
        # Determine if memory should be short-term or long-term
        important_keywords = ['name is', 'i am', 'my goal', 'important', 'remember', 'always']

        if access_count > 3:
            return 'long_term'
        
        if any(keyword in content.lower() for keyword in important_keywords):
            return 'long_term'
        
        # Memories older than 7 days 
        try:
            created_date = datetime.fromisoformat(created_at)
            if (datetime.now() - created_date).days > 7 and access_count > 1:
                return 'long_term'
        except:
            pass
        
        return 'short_term'
    
    @staticmethod
    async def create_narrative_summary(user_id: str, workspace_id: str, memories: List[Dict]) -> str:
        # Create a narrative summary from atomic memories
        if not memories:
            return ""
        
        memory_texts = [m['content'] for m in memories[:10]] # Limit to 10 memories for summary

        prompt = f"""Create a brief narrative summary of these conversation memories:
        {json.dumps(memory_texts, indent=2)}
        
        Summarize the key themes, goals, and important information about the user in 3-4 sentences."""

        try:
            response = co.generate(
                model='command',
                prompt=prompt,
                max_tokens=150,
                temperature=0.3
            )
            return response.generations[0].text.strip()
        except:
            return "User has shared various information across conversations."
# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize Qdrant collections on startup"""
    await init_qdrant_collections()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "features": ["reranking", "narrative_summaries", "lt_st_memory", "qdrant", "auto_traits"],
        "version": "2.0"
    }

# Main memory storage endpoint with enhancements
@app.post("/memory")
async def store_memory(request: MemoryRequest, background_tasks: BackgroundTasks):
    """Enhanced memory storage with reasoning traces and auto-trait extraction"""
    try:
        memory_id = str(uuid.uuid4())
        memory_content = f"User: {request.message}\nAssistant: {request.response}"
        
        # Extract user traits
        traits = {}
        name = UserTraits.extract_name(request.message)
        if name:
            traits['name'] = name
        
        location = UserTraits.extract_location(request.message)
        if location:
            traits['location'] = location
        
        role = UserTraits.extract_role(request.message)
        if role:
            traits['role'] = role
        
        sentiment = await UserTraits.analyze_sentiment(request.message)
        
        # Update user traits if any found
        if traits or sentiment:
            background_tasks.add_task(update_user_traits, request.userId, traits, sentiment)
        
        # Get embedding
        response = co.embed(
            texts=[memory_content],
            model=EMBEDDING_MODEL,
            input_type='search_document'
        )
        embedding = response.embeddings[0]
        
        # Store in Qdrant
        qdrant_client.upsert(
            collection_name=MEMORIES_COLLECTION,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "user_id": request.userId,
                        "workspace_id": request.workspaceId,
                        "thread_id": request.threadId,
                        "content": memory_content,
                        "reasoning": request.reasoning,
                        "created_at": datetime.now().isoformat(),
                        "memory_type": "short_term"
                    }
                )
            ]
        )
        
        # Store in SQLite for quick access
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        metadata = {'traits': traits} if traits else {}
        
        c.execute('''
            INSERT INTO memories 
            (id, user_id, content, metadata, memory_type, workspace_id, thread_id, reasoning_trace, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory_id,
            request.userId,
            memory_content,
            json.dumps(metadata),
            "short_term",
            request.workspaceId,
            request.threadId,
            request.reasoning,
            np.array(embedding).astype(np.float32).tobytes()
        ))
        
        conn.commit()
        conn.close()
        
        # Schedule narrative summary update
        background_tasks.add_task(update_narrative_summary, request.userId, request.workspaceId)
        
        logger.info(f"Memory stored successfully with ID: {memory_id}")
        return {"success": True, "id": memory_id}
        
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Context fallback 
async def get_context_fallback(request: MemoryContextRequest):
    """Fallback to SQLite-based context retrieval when Qdrant fails"""
    try:
        logger.info("[Fallback] Using SQLite-based context retrieval")
        
        # Get query embedding
        response = co.embed(
            texts=[request.message],
            model=EMBEDDING_MODEL,
            input_type='search_query'
        )
        message_embedding = response.embeddings[0]
        
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        # Get all memories for user
        c.execute('''
            SELECT content, metadata, embedding, created_at, memory_type
            FROM memories
            WHERE user_id = ? AND workspace_id = ?
            ORDER BY created_at DESC
            LIMIT 50
        ''', (request.userId, request.workspaceId))
        
        memories = c.fetchall()
        
        relevant_memories = []
        for content, metadata, stored_embedding, created_at, memory_type in memories:
            if stored_embedding:
                stored_embedding = np.frombuffer(stored_embedding, dtype=np.float32)
                if stored_embedding.shape[0] == 1024:
                    # Calculate cosine similarity
                    similarity = np.dot(message_embedding, stored_embedding) / (
                        np.linalg.norm(message_embedding) * np.linalg.norm(stored_embedding)
                    )
                    if similarity > MEMORY_THRESHOLD:
                        relevant_memories.append({
                            'content': content,
                            'metadata': json.loads(metadata) if metadata else {},
                            'similarity': float(similarity),
                            'created_at': created_at,
                            'memory_type': memory_type
                        })
        
        conn.close()
        
        # Sort by similarity
        relevant_memories.sort(key=lambda x: x['similarity'], reverse=True)
        relevant_memories = relevant_memories[:MAX_MEMORIES]
        
        # Get user traits
        traits = await get_user_traits(request.userId)
        
        # Format context
        context_parts = []
        
        # Add user traits if available
        if traits:
            trait_summary = []
            if traits.get('name'):
                trait_summary.append(f"Name: {traits['name']}")
            if traits.get('role'):
                trait_summary.append(f"Role: {traits['role']}")
            if traits.get('location'):
                trait_summary.append(f"Location: {traits['location']}")
            
            if trait_summary:
                context_parts.append("User Information:\n" + "\n".join(trait_summary))
        
        # Add memories
        context_parts.append("\nRelevant Memories:")
        for i, m in enumerate(relevant_memories):
            context_parts.append(f"\nMemory {i + 1}: {m['content']}")
        
        return {
            'context': '\n'.join(context_parts),
            'memories_used': len(relevant_memories),
            'has_narrative': False,
            'user_traits': traits
        }
        
    except Exception as e:
        logger.error(f"Error in fallback context: {e}")
        return {
            'context': '', 
            'memories_used': 0,
            'has_narrative': False,
            'user_traits': {}
        }



@app.post("/thread/save")
async def save_thread_state(request: ThreadStateRequest):
    """Save thread state to prevent resets"""
    try:
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        # Check if thread exists
        c.execute('SELECT thread_id FROM thread_states WHERE thread_id = ?', (request.threadId,))
        exists = c.fetchone()
        
        messages_json = json.dumps(request.messages)
        
        if exists:
            # Update existing thread
            c.execute('''
                UPDATE thread_states
                SET messages = ?, updated_at = CURRENT_TIMESTAMP, title = ?
                WHERE thread_id = ?
            ''', (messages_json, request.title, request.threadId))
            logger.info(f"Updated existing thread: {request.threadId}")
        else:
            # Create new thread
            c.execute('''
                INSERT INTO thread_states (thread_id, workspace_id, user_id, title, messages)
                VALUES (?, ?, ?, ?, ?)
            ''', (request.threadId, request.workspaceId, request.userId, request.title, messages_json))
            logger.info(f"Created new thread: {request.threadId}")
        
        conn.commit()
        conn.close()
        
        return {"success": True, "thread_id": request.threadId}
        
    except Exception as e:
        logger.error(f"Error saving thread state: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# Enhanced context retrieval with reranking
@app.post("/context")
async def get_enhanced_context(request: MemoryContextRequest):
    """Get context with reranking and narrative summaries"""
    try:
        # First, try to get user traits for basic info
        traits = await get_user_traits(request.userId)
        
        # Handle direct name queries
        if any(phrase in request.message.lower() for phrase in ["what is my name", "what's my name", "who am i"]):
            if traits and traits.get('name'):
                return {
                    "context": f"User's name is {traits['name']}.",
                    "memories_used": 1,
                    "has_narrative": False,
                    "user_traits": traits
                }
        
        # Get query embedding
        response = co.embed(
            texts=[request.message],
            model=EMBEDDING_MODEL,
            input_type='search_query'
        )
        query_embedding = response.embeddings[0]
        
        # Search in Qdrant with proper error handling
        search_results = []
        try:
            search_results = qdrant_client.search(
                collection_name=MEMORIES_COLLECTION,
                query_vector=query_embedding,
                limit=30,  # Increased limit to get more memories
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id",
                            match=models.MatchValue(value=request.userId)
                        ),
                        models.FieldCondition(
                            key="workspace_id", 
                            match=models.MatchValue(value=request.workspaceId)
                        )
                    ]
                )
            )
        except Exception as qdrant_error:
            logger.error(f"Qdrant search error: {qdrant_error}")
            # Fall back to SQLite search if Qdrant fails
            return await get_context_fallback(request)
        
        # Get narrative summary
        narrative = await get_narrative_summary(request.userId, request.workspaceId)
        
        # Prepare candidates for reranking
        candidates = []
        for result in search_results:
            candidates.append({
                'content': result.payload['content'],
                'score': result.score,
                'memory_type': result.payload.get('memory_type', 'short_term'),
                'reasoning': result.payload.get('reasoning', ''),
                'created_at': result.payload.get('created_at', ''),
                'id': result.id
            })
        
        # For "tell me about me" queries, include ALL memories regardless of similarity
        if any(phrase in request.message.lower() for phrase in ["tell me about me", "what do you know about me", "about me"]):
            # Get all user memories from SQLite for comprehensive response
            conn = sqlite3.connect('memories.db')
            c = conn.cursor()
            c.execute('''
                SELECT content, created_at, memory_type
                FROM memories
                WHERE user_id = ? AND workspace_id = ?
                ORDER BY created_at DESC
                LIMIT 20
            ''', (request.userId, request.workspaceId))
            
            all_memories = c.fetchall()
            conn.close()
            
            # Add these to candidates if not already present
            for content, created_at, memory_type in all_memories:
                if not any(c['content'] == content for c in candidates):
                    candidates.append({
                        'content': content,
                        'score': 0.5,  # Default score for non-vector matches
                        'memory_type': memory_type,
                        'reasoning': '',
                        'created_at': created_at,
                        'id': 'sqlite_' + str(hash(content))
                    })
        
        # Rerank using Cohere
        reranked_memories = []
        if candidates:
            try:
                # For "about me" queries, use all candidates
                if any(phrase in request.message.lower() for phrase in ["tell me about me", "what do you know about me", "about me"]):
                    # Skip reranking for "about me" queries - use all memories
                    reranked_memories = sorted(candidates, key=lambda x: x.get('created_at', ''), reverse=True)[:15]
                else:
                    # Normal reranking for specific queries
                    rerank_response = co.rerank(
                        model=RERANK_MODEL,
                        query=request.message,
                        documents=[c['content'] for c in candidates],
                        top_n=min(RERANK_TOP_N, len(candidates))
                    )
                    
                    for r in rerank_response:
                        idx = r.index
                        if r.relevance_score > MEMORY_THRESHOLD:
                            reranked_memories.append({
                                'content': candidates[idx]['content'],
                                'relevance_score': r.relevance_score,
                                'memory_type': candidates[idx]['memory_type'],
                                'reasoning': candidates[idx]['reasoning'],
                                'id': candidates[idx]['id']
                            })
            except Exception as e:
                logger.error(f"Reranking failed, using all candidates: {e}")
                # Use all candidates sorted by score
                reranked_memories = sorted(candidates, key=lambda x: x['score'], reverse=True)[:10]
        
        # Update access counts for used memories
        if reranked_memories:
            memory_ids = [m['id'] for m in reranked_memories if not m['id'].startswith('sqlite_')]
            if memory_ids:
                await update_memory_access(memory_ids)
        
        # Format context with better organization
        context_parts = []
        
        # Add user traits summary first
        if traits:
            trait_summary = []
            if traits.get('name'):
                trait_summary.append(f"Name: {traits['name']}")
            if traits.get('role'):
                trait_summary.append(f"Role: {traits['role']}")
            if traits.get('location'):
                trait_summary.append(f"Location: {traits['location']}")
            
            if trait_summary:
                context_parts.append("User Information:\n" + "\n".join(trait_summary))
        
        # Add narrative summary if available
        if narrative:
            context_parts.append(f"\nUser Profile Summary: {narrative}")
        
        # Extract and organize information from memories
        extracted_info = {
            'activities': [],
            'interests': [],
            'work': [],
            'personal': []
        }
        
        # Parse memories to extract structured information
        for mem in reranked_memories:
            content = mem['content'].lower()
            
            # Extract activities
            if any(word in content for word in ['walk', 'exercise', 'hobby', 'like to', 'enjoy']):
                if 'dog' in content:
                    extracted_info['activities'].append("Walks their dog")
                    
            # Extract interests
            if any(word in content for word in ['machine learning', 'ai', 'interested in', 'learning']):
                if 'machine learning' in content:
                    extracted_info['interests'].append("Machine learning")
                    
            # Extract work info
            if any(word in content for word in ['job', 'work', 'engineer', 'role']):
                if 'ai engineer' in content or 'engineer' in content:
                    extracted_info['work'].append("AI Engineer")
        
        # Add extracted information to context
        if extracted_info['activities']:
            context_parts.append(f"\nActivities: {', '.join(set(extracted_info['activities']))}")
        if extracted_info['interests']:
            context_parts.append(f"Interests: {', '.join(set(extracted_info['interests']))}")
        if extracted_info['work']:
            context_parts.append(f"Work: {', '.join(set(extracted_info['work']))}")
        
        # Add recent conversation memories
        context_parts.append("\nRecent Conversations:")
        for i, mem in enumerate(reranked_memories[:5]):
            context_parts.append(f"\nMemory {i+1}: {mem['content']}")
        
        return {
            'context': '\n'.join(context_parts),
            'memories_used': len(reranked_memories),
            'has_narrative': bool(narrative),
            'user_traits': traits or {}
        }
        
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        # Fallback to original SQLite-based search
        return await get_context_fallback(request)

@app.get("/thread/{thread_id}")
async def get_thread_state(thread_id: str):
    """Retrieve thread state"""
    try:
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT workspace_id, user_id, title, messages, created_at, updated_at
            FROM thread_states
            WHERE thread_id = ?
        ''', (thread_id,))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            return {
                "thread_id": thread_id,
                "workspace_id": result[0],
                "user_id": result[1],
                "title": result[2],
                "messages": json.loads(result[3]),
                "created_at": result[4],
                "updated_at": result[5]
            }
        else:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=404,
                content={"detail": "Thread not found"}
            )
            
    except Exception as e:
        logger.error(f"Error retrieving thread state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# User traits endpoints
@app.get("/user/traits/{user_id}")
async def get_user_traits_endpoint(user_id: str):
    """Get auto-extracted user traits"""
    traits = await get_user_traits(user_id)
    return {"user_id": user_id, "traits": traits}

@app.get("/user/goals/{user_id}")
async def get_user_goals(user_id: str):
    """Auto-summarize user goals from conversations"""
    try:
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT content FROM memories
            WHERE user_id = ? 
            AND (content LIKE '%goal%' OR content LIKE '%want%' OR content LIKE '%plan%')
            ORDER BY created_at DESC
            LIMIT 10
        ''', (user_id,))
        
        memories = [row[0] for row in c.fetchall()]
        conn.close()
        
        if not memories:
            return {"user_id": user_id, "goals": []}
        
        prompt = f"""Extract and summarize the user's goals from these conversations:
        {json.dumps(memories, indent=2)}
        
        List the main goals in bullet points:"""
        
        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=200,
            temperature=0.3
        )
        
        goals_text = response.generations[0].text.strip()
        goals = [g.strip() for g in goals_text.split('\n') if g.strip().startswith('-')]
        
        return {"user_id": user_id, "goals": goals}
        
    except Exception as e:
        logger.error(f"Error getting user goals: {e}")
        return {"user_id": user_id, "goals": []}

# Username endpoint (backward compatibility)
@app.get("/username")
async def get_username(userId: str, workspaceId: str):
    """Get username from traits"""
    traits = await get_user_traits(userId)
    return {"username": traits.get('name')}

# Helper functions
async def update_user_traits(user_id: str, traits: Dict, sentiment: str):
    """Update user traits in database"""
    conn = sqlite3.connect('memories.db')
    c = conn.cursor()
    
    c.execute('SELECT user_id FROM user_traits WHERE user_id = ?', (user_id,))
    exists = c.fetchone()
    
    if exists:
        update_parts = []
        params = []
        
        for key, value in traits.items():
            update_parts.append(f"{key} = ?")
            params.append(value)
        
        if sentiment:
            update_parts.append("sentiment_profile = ?")
            params.append(sentiment)
        
        if update_parts:
            update_parts.append("updated_at = CURRENT_TIMESTAMP")
            params.append(user_id)
            
            query = f"UPDATE user_traits SET {', '.join(update_parts)} WHERE user_id = ?"
            c.execute(query, params)
    else:
        c.execute('''
            INSERT INTO user_traits (user_id, name, location, role, sentiment_profile)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            traits.get('name'),
            traits.get('location'),
            traits.get('role'),
            sentiment
        ))
    
    conn.commit()
    conn.close()

async def update_narrative_summary(user_id: str, workspace_id: str):
    """Update narrative summary for user"""
    try:
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT id, content FROM memories
            WHERE user_id = ? AND workspace_id = ?
            ORDER BY created_at DESC
            LIMIT 20
        ''', (user_id, workspace_id))
        
        memories = [{'id': row[0], 'content': row[1]} for row in c.fetchall()]
        
        if len(memories) >= 5:
            summary = await MemoryManager.create_narrative_summary(user_id, workspace_id, memories)
            
            if summary:
                summary_id = str(uuid.uuid4())
                linked_ids = json.dumps([m['id'] for m in memories[:10]])
                
                c.execute('''
                    INSERT INTO narrative_summaries (id, user_id, workspace_id, summary, linked_memories)
                    VALUES (?, ?, ?, ?, ?)
                ''', (summary_id, user_id, workspace_id, summary, linked_ids))
                
                conn.commit()
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error updating narrative summary: {e}")

async def get_narrative_summary(user_id: str, workspace_id: str) -> Optional[str]:
    """Get latest narrative summary"""
    conn = sqlite3.connect('memories.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT summary FROM narrative_summaries
        WHERE user_id = ? AND workspace_id = ?
        ORDER BY created_at DESC
        LIMIT 1
    ''', (user_id, workspace_id))
    
    result = c.fetchone()
    conn.close()
    
    return result[0] if result else None

async def update_memory_access(memory_ids: List[str]):
    """Update access count and time for memories"""
    conn = sqlite3.connect('memories.db')
    c = conn.cursor()
    
    for memory_id in memory_ids:
        c.execute('''
            UPDATE memories
            SET access_count = access_count + 1,
                accessed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (memory_id,))
        
        # Check if should promote to long-term
        c.execute('''
            SELECT content, access_count, created_at
            FROM memories
            WHERE id = ?
        ''', (memory_id,))
        
        result = c.fetchone()
        if result:
            content, access_count, created_at = result
            new_type = MemoryManager.determine_memory_type(content, access_count, created_at)
            
            c.execute('UPDATE memories SET memory_type = ? WHERE id = ?', (new_type, memory_id))
            
            # Update in Qdrant too
            try:
                qdrant_client.set_payload(
                    collection_name=MEMORIES_COLLECTION,
                    payload={"memory_type": new_type},
                    points=[memory_id]
                )
            except:
                pass
    
    conn.commit()
    conn.close()

async def get_user_traits(user_id: str) -> Dict:
    """Get user traits"""
    conn = sqlite3.connect('memories.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT name, location, timezone, role, preferences, goals, sentiment_profile
        FROM user_traits
        WHERE user_id = ?
    ''', (user_id,))
    
    result = c.fetchone()
    conn.close()
    
    if result:
        return {
            'name': result[0],
            'location': result[1],
            'timezone': result[2],
            'role': result[3],
            'preferences': json.loads(result[4]) if result[4] else None,
            'goals': json.loads(result[5]) if result[5] else None,
            'sentiment': result[6]
        }
    return {}

# Backward compatibility endpoints
@app.post("/search")
async def search_memories(request: Dict[str, Any]):
    """Search memories - backward compatibility"""
    try:
        # Convert to context request
        context_req = MemoryContextRequest(
            userId=request.get('userId'),
            message=request.get('query'),
            workspaceId=request.get('workspaceId')
        )
        result = await get_enhanced_context(context_req)
        
        # Convert response format
        memories = []
        if result['memories_used'] > 0:
            # Parse memories from context
            context_parts = result['context'].split('\n\n')
            for part in context_parts:
                if part.startswith('Memory'):
                    memories.append({
                        'content': part,
                        'metadata': {},
                        'similarity': 0.8
                    })
        
        return memories[:request.get('limit', 10)]
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return []

@app.get("/memories")
async def get_memories(userId: str, workspaceId: str):
    """Get all memories for a user - backward compatibility"""
    try:
        conn = sqlite3.connect('memories.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT content, metadata, created_at
            FROM memories
            WHERE user_id = ? AND workspace_id = ?
            ORDER BY created_at DESC
        ''', (userId, workspaceId))
        
        memories = c.fetchall()
        
        results = [{
            'content': memory[0],
            'metadata': json.loads(memory[1]) if memory[1] else {},
            'created_at': memory[2]
        } for memory in memories]
        
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"Error getting memories: {e}")
        return []

@app.post("/memory/add")
async def add_memory(request: Dict[str, Any]):
    """Add memory - backward compatibility"""
    try:
        memory_req = MemoryRequest(
            userId=request.get('userId'),
            message=request.get('content', ''),
            response='',
            workspaceId=request.get('workspaceId'),
            threadId=None,
            reasoning=None
        )
        
        # Create a mock background tasks
        background_tasks = BackgroundTasks()
        result = await store_memory(memory_req, background_tasks)
        return result
        
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    logger.info("Starting enhanced memory server")
    uvicorn.run(app, host=HOST, port=PORT)