const express = require("express");

const router = express.Router();

//const { getSummary } = require("../controllers/dashboardController");

const {
  getSummary,
  getCategoryStats,
  getReturnTrends,
} = require("../controllers/dashboardController");

router.get("/summary", getSummary);
router.get("/categories", getCategoryStats);
router.get("/trends", getReturnTrends);
module.exports = router;