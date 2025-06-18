import axios from "axios";

const MEMORY_SERVICE_URL = "http://localhost:5002"; // ✅ Or proxy path if set

const client = axios.create({
  baseURL: MEMORY_SERVICE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

const MemoryIntegration = {
  async getUsername(userId, workspaceId) {
    try {
      const res = await client.get("/username", {
        params: { userId, workspaceId }
      });
      return res.data.username; // ✅ Return just string
    } catch (err) {
      console.error("Error fetching username:", err);
      return null;
    }
  }
};

export default MemoryIntegration;