const fetch = require('node-fetch');

class MemoryIntegration {
    constructor() {
        this.baseUrl = process.env.PYTHON_MEMORY_URL || 'http://localhost:5002';
        console.log(`[MemoryIntegration] Initialized with base URL: ${this.baseUrl}`);

    }
    async storeChatMemory(userId, message, response, workspaceId, threadId = null, reasoning = null) {
        try {
            console.log(`[MemoryIntegration] Storing memory for user ${userId} in workspace ${workspaceId}`); 
            const res = await fetch(`${this.baseUrl}/memory`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    userId,
                    message,
                    response,
                    workspaceId,
                    threadId,
                    reasoning
                }),
            });

            if (!res.ok) {
                const errorText = await res.text();
                console.error(`[MemoryIntegration] Error response:`, errorText);
                throw new Error(`Error storing chat memory: ${res.statusText}`);
            }
            const result = await res.json();
            console.log(`Memory stored with ID: ${result.id}`);
            return result;

        } catch (error) {
            console.error('[MemoryIntegration] Failed to store chat memory:', error);
            throw error; // Re-throw the error for further handling
        }
    }
    // Get enhanced chat context - reranking and narrative summaries
    async getChatContext(userId, message, workspaceId) {
        try {
            console.log(`[MemoryIntegration] Getting context for user ${userId} in workspace ${workspaceId}`);
            const res = await fetch(`${this.baseUrl}/context`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    userId,
                    message,
                    workspaceId
                })
            });
            if (!res.ok) {
                const errorText = await res.text();
                console.error(`[MemoryIntegration] Context retrieval failed: ${res.statusText}`);
                console.error('[MemoryIntegration] Error response:', errorText);
                if (errorText.includes('Index required but not found')) {
                    console.error('[MemoryIntegration] Qdrant index missing! Run fix_qdrant_indexes.py');
                }
                return {context: '', memories_used: 0};
            }
            const result = await res.json();
            console.log('[MemoryIntegration] context retrieved:', {
                memories_used: result.memories_used,
                has_narrative: result.has_narrative,
                user_traits: result.user_traits,
                context_preview: result.context ? result.context.substring(0, 100) + '...' : 'No context'
                })
            return {
                context: result.context || '',
                memories_used: result.memories_used || 0,
                has_narrative: result.has_narrative || false,
                user_traits: result.user_traits || {}
            };

        }  catch (error) {
            console.error('[MemoryIntegration] Failed to get context:', error);
            return {
                context: '', memories_used: 0
            };
        }
        
    }
    // Save thread state to prevent resets
    async saveThreadState(threadId, workspaceId, userId, messages, title =null) {
        try {
            console.log(`[MemoryIntegration] Saving thread state for ${threadId}`);
            const res = await fetch(`${this.baseUrl}/thread/save`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    threadId,
                    workspaceId,
                    userId,
                    title,
                    messages
                })
            });
            if(!res.ok) {
                const errorText = await res.text();
                console.error(`[MemoryIntegration] Thread Save error:`, errorText);
                throw new Error(`Thread save failed: ${res.statusText}`);
            }
            const result = await res.json();
            console.log(`[MemoryIntegration] Thread state saved for ${threadId}`);
            return result;

        } catch (error) {
            console.error('[MemoryIntegration] Failed to save thread state:', error);
            return null; // Re-throw for further handling
        }
}
// Load thread state
async loadThreadState(threadId) {
        try {
            console.log(`[MemoryIntegration] loading thread state for ${threadId}`);
            const res = await fetch(`${this.baseUrl}/thread/${threadId}`);
            
            if (!res.ok) {
                if (res.status === 404) {
                    console.log(`[MemoryIntegration] Thread ${threadId} not found`)
                    return null; // Thread not found
                }
                throw new Error(`Thread load failed: ${res.statusText}`);
            }
            
            const result = await res.json();
            console.log(`[MemoryIntegration] Thread state loaded`)
            return result;
        } catch (error) {
            console.error('Failed to load thread state:', error);
            return null;
        }
    }

    /**
     * Get user traits
     */
    async getUserTraits(userId) {
        try {
            console.log(`[MemoryIntegration] Getting traits for user ${userId}`);
            const res = await fetch(`${this.baseUrl}/user/traits/${userId}`);
            
            if (!res.ok) {
                return {};
            }

            const result = await res.json();
            return result.traits || {};
        } catch (error) {
            console.error('[MemoryIntegraiton] Failed to get user traits:', error);
            return {};
        }
    }

    /**
     * Get user goals
     */
    async getUserGoals(userId) {
        try {
            const res = await fetch(`${this.baseUrl}/user/goals/${userId}`);
            
            if (!res.ok) {
                return [];
            }

            const result = await res.json();
            return result.goals || [];
        } catch (error) {
            console.error('[MemoryIntegration] Failed to get user goals:', error);
            return [];
        }
    }

    /**
     * Get username (backward compatibility)
     */
    async getUsername(userId, workspaceId) {
        try {
            const res = await fetch(`${this.baseUrl}/username?userId=${userId}&workspaceId=${workspaceId}`);
            
            if (!res.ok) {
                return null;
            }

            const result = await res.json();
            return result.username;
        } catch (error) {
            console.error('[MemoryIntegration] Failed to get username:', error);
            return null;
        }
    }

    /**
     * Search memories (backward compatibility)
     */
    async searchMemories(userId, query, workspaceId, limit = 10) {
        try {
            console.log(`[MemoryIntegration] Searching memories for query: "${query}"`);
            const res = await fetch(`${this.baseUrl}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userId,
                    query,
                    workspaceId,
                    searchType: 'hybrid',
                    limit
                })
            });

            if (!res.ok) {
                console.error('[MemoryIntegration] Search failed:', res.statusText);
                return [];
            }

            const results = await res.json();
            console.log(`[MemoryIntegration] Found ${results.length} memories`);
            return results;
        } catch (error) {
            console.error('Failed to search memories:', error);
            return [];
        }
    }

    /**
     * Get all memories for a user
     */
    async getMemories(userId, workspaceId) {
        try {
            const res = await fetch(`${this.baseUrl}/memories?userId=${userId}&workspaceId=${workspaceId}`);
            
            if (!res.ok) {
                return [];
            }

            return await res.json();
        } catch (error) {
            console.error('Failed to get memories:', error);
            return [];
        }
    }

    /**
     * Add a memory directly
     */
    async addMemory(userId, content, workspaceId, metadata = {}) {
        try {
            const res = await fetch(`${this.baseUrl}/memory/add`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userId,
                    content,
                    workspaceId,
                    metadata
                })
            });

            if (!res.ok) {
                throw new Error(`Add memory failed: ${res.statusText}`);
            }

            return await res.json();
        } catch (error) {
            console.error('Failed to add memory:', error);
            throw error;
        }
    }
    async healthCheck() {
        try {
            const res = await fetch(`${this.baseUrl}/health`);
            if (!res.ok) {
                return false;
            }
            const data = await res.json();
            console.log(`[MemoryIntegration] Health check:`, data);
            return data.status === 'healthy';
        } catch (error) {
            console.error('[MemoryIntegration] Health check failed:', error);
            return false;
        }
    }
}

// Export singleton instance
module.exports = new MemoryIntegration();