import MemoryModel from "../models/memory.js";
import fs from "fs";
import path from "path";

// Load config - current working directory
const configPath = path.join(process.cwd(), "mem0config.json");
const rawConfig = fs.readFileSync(configPath, "utf-8");
const useConfig = JSON.parse(rawConfig);

class MemoryService {
    constructor() {
        // Init
        this.memoryModel = new MemoryModel(useConfig);
    }

    // Handles creating memory request
    async createMemory(req, res) {
        try {
            const {text, metadata = {} } = req.body;
            
            if (!text || typeof text !== "string") {
                return res.status(400).json({ success: false, error: "Field 'text' is required and must be a string." });
            }
            
            // Attach server-side metadata
            const enrichedMetadata = {
                ...metadata,
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString(),
            };
            
            // Call the model to store
            const memoryRecord = await this.memoryModel.createMemory({
                text,
                metadata: enrichedMetadata
            });
            return res.status(201).json({ success: true, data: memoryRecord, message: "Memory created successfully." });
            
        } catch (error) {
            console.error("MemoryService.createMemory error:", error);
            return res.status(500).json({success: false, error: "Internal server error"});
        }
    }

    // Handles getting memory request
    async getMemory(req, res) {
        try {
            const { query, topK } = req.query;
            
            if(!query || typeof query !== "string") {
                return res.status(400).json({ success: false, error: "Field 'query' is required and must be a string."});  
            }
            
            const k = parseInt(topK) || 5; // Default to 5 if not provided or invalid
            
            const results = await this.memoryModel.getMemory({
                query,
                topK: k
            });
            
            return res.status(200).json({ success: true, data: results, message: "Memory retrieved successfully." });
        } catch (err) {
            console.error("MemoryService.getMemory error:", err);
            return res.status(500).json({ success: false, error: "Internal server error"});
        }
    }

    // Handles removing memory request
    async deleteMemory(req, res) {
        try {
            const { id } = req.params;
            
            if(!id) {
                return res.status(400).json({ success: false, error: "Memory field 'id' is required."});
            }
            const result = await this.memoryModel.deleteMemory({ id});
            
            return res.status(200).json({ success: true, data: result, message: "Memory deleted successfully." });
            
        } catch (err) {
            console.error("MemoryService.deleteMemory error:", err);
            return res.status(500).json({ success: false, error: "Internal server error"});
        }
    }
}

export default MemoryService;