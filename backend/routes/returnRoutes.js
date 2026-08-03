const express = require("express");

const router = express.Router();

const { createReturn } = require("../controllers/returnController");

router.post("/", createReturn);

module.exports = router;