const axios = require("axios");

// CONFIG

const MEMORY_SERVICE_URL = process.env.MEMORY_SERVICE_URL || "http://localhost:5002";

class MemoryIntegration {
    constructor() {
        this.apiUrl = MEMORY_SERVICE_URL;
        this.client = axios.create({
            baseURL: this.apiUrl,
            timeout: 10000, // 10 seconds
            headers: {
                "Content-Type": "application/json"
            }
        });
    }
    extractUsername(text) {
        const regex = /(my name is|I'm|I am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i;
        const match = text.match(regex);
        return match ? match[2] : null;
    }
    
    async getChatContext(userId, message, workspaceId) {
        try {
            const response = await this.client.post("/context", {
                userId,
                message,
                workspaceId
            });
            return response.data;
        } catch (error) {
            console.error("Error getting chat context:", error);
            return { context: "", memories_used: 0 };
        }
    }

    async storeChatMemory(userId, message, responseText, workspaceId, threadId = null) {
        try {
            const username = this.extractUsername(message)
            const memory_content = `User: ${message}\nAssistant: ${responseText}`;
            const metadata = {
                type: "chat",
                ...(username && { username }) 
            };

            await this.client.post("/memory/add", {
                userId,
                message,
                content: memory_content,
                metadata,
                workspaceId,
                threadId,
            });
            
            return true;
        } catch (error) {
            console.error("Error storing chat memory:", error);
            return false;
        }
    }

    async searchMemories(userId, query, options = {}) {
        try {
            const response = await this.client.post("/search", {
                userId,
                query,
                ...options
            });
            return response.data;
        } catch (error) {
            console.error("Error searching memories:", error);
            return [];
        }
    }

    async getUserMemories(userId, workspaceId) {
        try {
            const response = await this.client.get("/memories", {
                params: { userId, workspaceId }
            });
            return response.data;
        } catch (error) {
            console.error("Error getting user memories:", error);
            return [];
        }
    }
    async getUsername(userId, workspaceId) {
    try {
      const res = await this.client.get("/username", {
        params: { userId, workspaceId }
      });
      return res.data.username;
    } catch (err) {
      console.error("Error fetching username:", err);
      return null;
    }
  }

    async addMemory(userId, content, metadata = {}, workspaceId) {
        try {
            const response = await this.client.post("/memory/add", {
                userId,
                content,
                metadata,
                workspaceId
            });
            return response.data;
        } catch (error) {
            console.error("Error adding memory:", error);
            return null;
        }
    }

    // Check if memory service is healthy
    async isHealthy() {
        try {
            const response = await this.client.get('/health');
            return response.data;
        } catch (error) {
            console.error("Memory service health check error:", error.message);
            return { status: "unhealthy", error: error.message };
        }
    }
}

module.exports = new MemoryIntegration();
