const Customer = require("../models/Customer");
const Return = require("../models/Return");

const createReturn = async (req, res) => {
  try {
    const {
      customer,
      orderId,
      category,
      reason,
      daysToReturn,
      customerRating,
      mismatchFlag,
    } = req.body;

    // Check if customer exists
    const existingCustomer = await Customer.findById(customer);

    if (!existingCustomer) {
      return res.status(404).json({
        success: false,
        message: "Customer not found",
      });
    }

    // Save return
    const newReturn = await Return.create({
      customer,
      orderId,
      category,
      reason,
      daysToReturn,
      customerRating,
      mismatchFlag,
    });

    res.status(201).json({
      success: true,
      message: "Return created successfully",
      data: newReturn,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

module.exports = {
  createReturn,
};