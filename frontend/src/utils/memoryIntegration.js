import axios from "axios";

const MEMORY_SERVICE_URL = "http://localhost:5002";

const client = axios.create({
  baseURL: MEMORY_SERVICE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

const MemoryIntegration = {
  /**
   * Get username for a user
   */
  async getUsername(userId, workspaceId) {
    try {
      const res = await client.get("/username", {
        params: { userId, workspaceId }
      });
      return res.data.username;
    } catch (err) {
      console.error("Error fetching username:", err);
      return null;
    }
  },

  /**
   * Save thread state to prevent resets
   * CRITICAL: Call this after every message exchange
   */
  async saveThreadState(threadId, workspaceId, userId, messages, title = null) {
    try {
      const res = await client.post("/thread/save", {
        threadId: String(threadId),
        workspaceId: String(workspaceId),
        userId: String(userId),
        title,
        messages
      });
      console.log(`Thread state saved: ${threadId}`);
      return res.data;
    } catch (err) {
      console.error("Error saving thread state:", err);
      return null;
    }
  },

  /**
   * Load thread state
   * CRITICAL: Call this when loading/switching threads
   */
  async loadThreadState(threadId) {
    try {
      const res = await client.get(`/thread/${threadId}`);
      console.log(`Thread state loaded: ${threadId}`);
      return res.data;
    } catch (err) {
      if (err.response?.status === 404) {
        console.log(`Thread ${threadId} not found in memory`);
        return null;
      }
      console.error("Error loading thread state:", err);
      return null;
    }
  },

  /**
   * Get user traits (name, location, role, etc.)
   */
  async getUserTraits(userId) {
    try {
      const res = await client.get(`/user/traits/${userId}`);
      return res.data.traits || {};
    } catch (err) {
      console.error("Error fetching user traits:", err);
      return {};
    }
  },

  /**
   * Get user goals
   */
  async getUserGoals(userId) {
    try {
      const res = await client.get(`/user/goals/${userId}`);
      return res.data.goals || [];
    } catch (err) {
      console.error("Error fetching user goals:", err);
      return [];
    }
  },

  /**
   * Get all memories for a user
   */
  async getMemories(userId, workspaceId) {
    try {
      const res = await client.get("/memories", {
        params: { userId, workspaceId }
      });
      return res.data || [];
    } catch (err) {
      console.error("Error fetching memories:", err);
      return [];
    }
  },

  /**
   * Search memories
   */
  async searchMemories(userId, query, workspaceId, limit = 10) {
    try {
      const res = await client.post("/search", {
        userId: String(userId),
        query,
        workspaceId: String(workspaceId),
        limit
      });
      return res.data || [];
    } catch (err) {
      console.error("Error searching memories:", err);
      return [];
    }
  }
};

export default MemoryIntegration;