const express = require('express');
const fs = require('fs');
const path = require('path');
const { MongoClient } = require('mongodb');
const axios = require('axios');
const bodyParser = require('body-parser');
const crypto = require('crypto');
const http = require('http');
const socketIo = require('socket.io');
const { body, validationResult } = require('express-validator');  // Add validation



const app = express();
const server = http.createServer(app);
const io = socketIo(server);
const port = 3000;
const mongoUrl = "mongodb://localhost:27017/";
const dbName = 'honeypot_monitoring';
const collectionName = 'visitor_logs';
const cors = require('cors');

let dbClient;

// Middleware for JSON parsing and URL encoding
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

// ✅ Serve static files (Prevents Directory Exposure)
app.use(express.static(path.join(__dirname, 'public')));

// ✅ Generate **Random Path + Secure Token** for Dashboard
const randomPath = crypto.randomBytes(8).toString('hex');
const token = crypto.randomBytes(16).toString('hex');
console.log(`🚀 Dashboard: http://localhost:${port}/dashboard-${randomPath}?token=${token}`);

// ✅ Middleware to Verify Token for Dashboard
const verifyToken = (req, res, next) => {
    const userToken = req.query.token;
    if (!userToken || userToken !== token) {
        return res.status(403).send('❌ Forbidden: Invalid token');
    }
    next();
};

const connectMongo = async () => {
    try {
        dbClient = new MongoClient(mongoUrl, { useNewUrlParser: true, useUnifiedTopology: true });
        await dbClient.connect();
        console.log('✅ Connected to MongoDB');
    } catch (error) {
        console.error('❌ MongoDB Connection Error:', error);
    }
};
connectMongo();


// ✅ Secure Dashboard Route
app.get(`/dashboard-${randomPath}`, verifyToken, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'dashboard.html'));
});


// ✅ Visitor Logging API with validation
app.post('/log', [
    body('real_ip').isIP().withMessage('Invalid IP address'),
    body('timezone').isString().withMessage('Invalid timezone'),
    // Add more validation rules if needed
], async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({ errors: errors.array() });
    }

    let ip = req.body.real_ip || req.socket.remoteAddress;
    let timezone = req.body.timezone;
    let geoLocation = { lat: 'Unknown', lon: 'Unknown' };

    try {
        if (!ip.includes("Unknown (Tor Hidden Service)")) {
            const geoData = await axios.get(`https://ipinfo.io/${ip}/json`);
            if (geoData.data.loc) {
                const loc = geoData.data.loc.split(',');
                geoLocation = { lat: parseFloat(loc[0]), lon: parseFloat(loc[1]) };
            }
            if (timezone === "UTC" || timezone === "Etc/UTC") {
                timezone = geoData.data.timezone || "Unknown";
            }
        }
    } catch (error) {
        console.error("❌ Error fetching geolocation/timezone:", error);
    }

    const logEntry = {
        ip,
        userAgent: req.get('User-Agent'),
        screen_size: req.body.screen_size,
        timezone,
        language: req.body.language,
        platform: req.body.platform,
        cpu_cores: req.body.cpu_cores,
        touch_support: req.body.touch_support,
        memory: req.body.memory,
        timestamp: new Date(),
        location: geoLocation,
    };

    console.log(`📡 Visitor Logged: ${JSON.stringify(logEntry)}`);

    // ✅ Append to logs.txt (Create if missing)
    const logFilePath = path.join(__dirname, 'logs.txt');
    fs.appendFileSync(logFilePath, JSON.stringify(logEntry) + '\n', 'utf8');

    try {
        // ✅ Save to MongoDB
        const db = dbClient.db(dbName);
        const collection = db.collection(collectionName);
        await collection.insertOne(logEntry);
    } catch (error) {
        console.error('❌ Error saving to MongoDB:', error);
    }

    // ✅ Broadcast to Dashboard (Real-Time Update)
    io.emit('newVisitor', logEntry);

    res.status(200).send('Visitor data logged successfully');
});

// ✅ Fetch Visitor Data for Dashboard
app.get('/getVisitorData', verifyToken, async (req, res) => {
    try {
        const db = dbClient.db(dbName);
        const collection = db.collection(collectionName);
        const visitors = await collection.find({}).toArray();

        console.log("📡 Sending Visitor Data:", visitors); // Debugging
        res.json(visitors);
    } catch (error) {
        console.error('❌ Error fetching visitor data:', error);
        res.status(500).send('❌ Error fetching data');
    }
});

app.get('/getDashboardUrl', (req, res) => {
    res.json({ 
        url: `http://localhost:${port}/dashboard-${randomPath}`,
        token: token
    });
});

app.use(cors({
    origin: 'http://localhost:3000', // Update with your React app's origin
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type', 'Authorization']
  }));


// ✅ Start Server
server.listen(port, () => console.log(`✅ Server running at http://localhost:${port}`));
