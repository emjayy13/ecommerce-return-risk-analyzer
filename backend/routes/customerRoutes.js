const express = require("express");

const router = express.Router();

const {
  createCustomer,
  getAllCustomers,
  getCustomerById,
} = require("../controllers/customerController");

router.route("/")
  .post(createCustomer)
  .get(getAllCustomers);

router.get("/:id", getCustomerById);
router.post("/", createCustomer);

module.exports = router;