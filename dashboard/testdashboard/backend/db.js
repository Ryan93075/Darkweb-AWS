const express = require("express");
const { MongoClient } = require("mongodb");
const cors = require("cors");

const app = express();
const PORT = 5000;
const uri = "mongodb://localhost:27017"; // Update if necessary

app.use(cors()); // Enable CORS to allow frontend requests

async function fetchThreatData() {
    const client = new MongoClient(uri);
    try {
        await client.connect();
        const db = client.db("local"); 
        const collection = db.collection("test_1"); 
        return await collection.find().toArray(); 
    } catch (error) {
        console.error("❌ Error fetching data:", error);
        return [];
    } finally {
        await client.close();
    }
}

// API Endpoint to fetch threats
app.get("/api/threats", async (req, res) => {
    const threats = await fetchThreatData();
    res.json(threats);
});

app.listen(PORT, () => {
    console.log(`✅ Server running on http://localhost:${PORT}`);
});
