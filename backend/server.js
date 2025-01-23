const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');  // Import axios

const app = express();
app.use(cors());
app.use(bodyParser.json());

app.post('/generate', async (req, res) => {
    const prompt = req.body.prompt;
    try {
        // Call the Flask API
        const aiResponse = await axios.post('http://localhost:5000/generate', { prompt });
        res.json(aiResponse.data);  // Forward AI response to the frontend
    } catch (err) {
        res.status(500).json({ error: 'AI server error' });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
