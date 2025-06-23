# Notes
**Author** <Cade Miller>
**Start Date** <2025-06-03>

# Added to docker/.env 
- <SERVER_PORT, LLM_PROVIDER, OLLAMA_BASE_URL>

# Full rebranded Ruya (scripts below):
# Usage: bash rename_brand.sh
# Description: Replaces all instances of Ruya-LLM branding with RuyaLLM.

find . -type f \
  -not -path '*/node_modules/*' \
  -not -path '*/.git/*' \
  -not -path '*.png' \
  -not -path '*.jpg' \
  -not -path '*.jpeg' \
  -not -path '*.ico' \
  -not -path '*.svg' \
  -exec grep -Il '.' {} \; | while read -r file; do
    echo "Updating: $file"
    sed -i '' -e 's/ruya-llm/ruya-llm/g' "$file"
    sed -i '' -e 's/Ruya-LLM/Ruya-LLM/g' "$file"
    sed -i '' -e 's/RuyaLLM/RuyaLLM/g' "$file"
done


# Summary (06-03-25/06-04-25)
- UI/UX: 
    - Changed background surrounding the chatbot and interior to be a dark black via frontend/index.css
    - Added a linear gradient border & a glow that reflects the same color surrounding the chatbot, purple -> turqouise. via frontend/index.css
    - Modified Chat history font to white and added thicknes to divisor via src/components/Workspace/ChatContent/ChatHistory/index.js
    - Modified border between Send Chat messsage and icons for upload via frontend/index.css
    - Created a purple accent around the send message via frontend/index.css
    - Isolated the sidepanels text via frontend/index.css
    - Created four color schemes for Ruya (ruyaPurple, ruyaTurqouise, ruyaPanel, ruyaBlack) via frontend/tailwind.config
    - Isolated chat text with a --theme-chat-input-text via frontend/tailwind.config
    - Modified dropdown arrow to see most recent chatHistory and added a pulse animation via frontend/index.css
    - Added the gradient border & glow effect to src/components/WorkspaceChat/ChatContainer/index.js
    - Added a new userIcon stylisticly beautiful src/components/UserIcon
    - In big bold letters new thread font when enter page with ruyaPurple, White, ruyaTurqouise and changed to Ruya based src/components/workspaceChat/index.js
# Scope
- General Changes
    - Include: Branding, Tailwind customization, chat history and chat enhancements.
    
# Tailwind changes
- animation: pulse
- chat-input: text
- bg: primary
- colors: ruyaPurple, ruyaTurqouise, ruyaPanel, ruyaBlack
- Allowed files: "./src/**/*.{js,jsx,ts,tsx}",

# Additional Detail/Changes
- Updated App.jsx: added color scheme
- Updated src/components/Workspace/ChatContent/ChatHistory/PromptInput: border style (Above AttachmentManager)
- Updated index.css @layer components (gradient-border)



# Summary (6-20-25)
- Backend server/...:
    - memory/memoryIntegration:
        - StoreChatMemory(): saves a coversation exchange to memory
            - Parameters: userId, message, response, workspaceId, threadId, reasoning
            - Returns: which conversation thread

        - getChatContext(): Gets relevant past conversations to help AI understand context
            - Parameters: userId, message, workspaceId
            - Returns: Context, memories_used, has_narrative, user_traits
            - Special Features: Uses reranking for better relevance, Creates narrative summaries, handles missing indexes 

        - saveThreadState(): Saves the current state of a conversation thread
            - Parameters: threadId, workspaceId, userId, messages, title
            - Use case: prevent conversation resets when users return

        - loadThreadState(): Retrieves a previously saved conversation thread
            - Parameters: ThreadId
            - Returns: Thread data or null if not found
        
        - getUserTraits(): Gets Personality traits and preferences for a user
            - Parameters: userId
            - Returns: Object with user traits or {}

        - getUserGoals(): Gets user's goals or objectives
            - Parameters: userId
            - Returns: Array of goals or []
        
        - getUsername(): Gets the username for backward compatibility
            - Parameters: userId, workspaceId
            - Returns: Username string or null
        
        - searchMemories(): Searches through stored memories
            - Parameters: userId, query, workspaceId, limit (default=10)
            - Special Features: Use hybrid search (combines lexical & Semantic)
            - Returns: Array of matching memories or []
        
        - getMemories(): Gets all memories for a user
            - Parameters: userId, workspaceId
            - Returns: Array of all memories

        - addMemory(): Directly adds a memory without a conversation
            - Parameters: userId, content, workspaceId, metadata
            - Use Case: Adds notes / important information outside of chat
        
        - healthCheck(): Checks if the memory service is runningy.
            - Returns: Boolean (true if health, false otherwise)
            - Checks: Service equals healthy
    
    - python-memory/
        - memory_server
        - Key Features: Hybrid Search, Memory Reranking, Narrative Summaries, Short/Long-term Memory, User Trait Extraction, Thread Persisten, Qdrant Integration, SQLite Backup

        - Tehcnology Stack: FastAPI, Cohere, Qdrant, SQLite, Python Lib: Pydantic, numpy, asyncio

        - SqLite Tables
            - Memories: 
                - Stores all conversation memories
                - Fields: id, user_id, content, metadata, memory_type, workspace_id, thread_id, reasoning_trace, embedding, timestamps
            
            - user_traits
                - Stores extracted user information
                - Fields: user_id, name, location, timezone, role, preferences, goals, sentiment_profile

            - thread_states
                - Saves conversation threads
                - Fields: thread_id, workspace_id, user_id, title, summary, messages, timestamps

            - narrative_summaries
                - Stores AI-generated Summaries
                - Fields: id, user_id, workspace_id, summary, linked_memories
            
            - Qdrant collections
                - ruya_memories: Vector storage for memory embeddings
                - ruya_threads: Thread embeddings
                - ruya_summaries: summary embeddings
            
            - Main API endpoint

            - Health & status

                - Get /health
                    - Check if server is running
                    - Returns status and feature list
            
            - Memory Operation

                - POST/memory: store a conversation exchange
                    - Input: userId, message, response, workspaceId, threadId, reasoning
                    - Extracts user traits from message
                    - Creates Embedding
                    - Stores in Qdrant and SQLite
                    - Schedule background tasks
                    - Returns Memory ID
                
                - POST/context: Get relevant context for AI responses
                    - Input: userId, message, workspaceId
                    - Searches similar memories
                    - Reranks results
                    - Includes user traits and summaries
                    - Handles special queries (name, about me)
                    - Return: Formatted context with metadata
            
            - Thread Management

                - POST/thread/save: Save conversation thread state
                    - Input: threadId, worspaceId, userId, messages, title
                    - Returns: success status
                
                - GET/thread/{thread_id}: Load a saved thread
                    - Returns: Thread data or 404 if not found

            - User Information

                - GET/user/traits/{user_id}: Get extracted user traits
                    - Returns: Name, location, role, preferences
                
                - GET/user/goals/{user_id}: Get AI-summarized user goals
                    - Returns: List of identified goals
                
                - GET/username: Get user's name
                    - Returns: Username if known
                
            - Search & Retrieval
                
                - POST/search: Search memories by query
                    - Input: userId, query, workspaceId, searchType, limit
                    - Returns: matching memories
                
                - Get/memories: Get all memories for a user
                    - Returns: List of all memories
                
                - POST/memory/Add: Add memory directly
                    - Input: userId, content, workspaceId, metadata
                    - Returns: memoryId
            
        - Helper Classes

            - UserTraits
                - extract_name(): Find user's name
                - extract_location(): Identifies location
                - extract_role(): Detects job/role
                - analyze_sentiment(): Basic Sentiment analysis

            - MemoryManager
                - determine_memory_type(): Decides short/long-term status
                - create_narrative_summary(): Generates story summaries

        - config
            - Mem0_config, Hybrid_search_config, Text_processing, Narrative_summaries, Trait_extraction_config, Thread_config, Rerank_config

        - hybrid_search

            - BM25, cosine similarity, Text Preprocessing, Score Normalization, Weighted combination, NLTK Integration, Stemming support

            - Core Methods

                - preprocess_text(): Cleans and prepares text for searching
                    - Converts to lowercase
                    - Removes Urls, special words
                    - Removes common words
                    - Removes very short words
                    - Optionally stems word to root form
                    - Returns: List of cleaned words

                - lecical_search(): Finds documents with matching keywords
                    - Parameters: query, documents, top_k
                    - Uses BM25 algo
                    - Scores based on keyword freq
                    - Considers document length
                    - Returns best keyword matches
                    - Returns: List of (document_index, score) pairs

                - Semantic_search(): Finds documents with similar meanings
                    - Parameters: query_embeddings, document_embeddings, top_k
                    - Compares vector similarities
                    - Uses cosine similarity metric
                    - Finds semantically related context
                    - Returns: List of (document_index, score) pairs
                
                - hybrid_search(): Combines both search methods for best results
                    - Parameters: query, query_embedding, documents, document_embeddings, top_k, alpha
                    - Runs both lexical and semantic search
                    - Normalizes scores to same scale
                    - Combines Scores with weighted average
                    - Sorts by combined score
                    - Returns: Ranked results with detailed scores

                - normalize_scores(): Converts scores to (0-1)
                    - Different search methods use different scales
                    - Method: min-max normalization
                    - Edge cases like identical scores
                
                - Cosine_simalarity(): Calculates similarity between vectors
                    - 1d & 2d vectors
                    - Multiple comparisons at once
                    - zero vectors safely
                    - Returns 0 = different & 1 = identical
                
                - get_search_enginer(): Gets or creates single search instance
                    - Resource efficient

    - utils/

        - stream.js
            - MemoryIntegration import 

            - getChatContext(): Gets relevant past conversations, adds memory context to available context texts, logs number of memories added 

            - storeChatMemory(): Only stores in "chat" mode

            - UserId: "persistent_user" - for each workspace keeps same user
