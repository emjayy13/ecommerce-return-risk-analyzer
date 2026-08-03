const axios = require("axios");

const ML_API_URL = process.env.ML_API_URL || "http://127.0.0.1:8000";

async function predictRisk(data) {
  const response = await axios.post(`${ML_API_URL}/predict`, data);
  return response.data;
}

module.exports = {
  predictRisk,
};