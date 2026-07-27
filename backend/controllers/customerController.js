const Customer = require("../models/Customer");
const Return = require("../models/Return");
const createCustomer = async (req, res) => {
  try {
    const customer = await Customer.create(req.body);

    res.status(201).json({
      success: true,
      message: "Customer created successfully",
      data: customer,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

const getAllCustomers = async (req, res) => {
  try {
    const { risk, returnRatio, category } = req.query;

    let query = {};

    // Filter by risk level
    if (risk) {
      query.riskLevel = risk;
    }

    let customers = await Customer.find(query).sort({
      riskScore: -1,
    });

    // Filter by return ratio
    if (returnRatio) {
      customers = customers.filter((customer) => {
        if (customer.totalOrders === 0) return false;

        return (
          customer.totalReturns / customer.totalOrders >=
          Number(returnRatio)
        );
      });
    }

    // Filter by category
    if (category) {
      const filteredCustomers = [];

      for (const customer of customers) {
        const hasCategory = await Return.exists({
          customer: customer._id,
          category,
        });

        if (hasCategory) {
          filteredCustomers.push(customer);
        }
      }

      customers = filteredCustomers;
    }

    res.status(200).json({
      success: true,
      count: customers.length,
      data: customers,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

const getCustomerById = async (req, res) => {
  try {
    const customer = await Customer.findById(req.params.id);

    if (!customer) {
      return res.status(404).json({
        success: false,
        message: "Customer not found",
      });
    }

    const returns = await Return.find({
      customer: customer._id,
    }).sort({
      returnedAt: -1,
    });

    res.status(200).json({
      success: true,
      data: {
        customer,
        returns,
      },
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

module.exports = {
  createCustomer,
  getAllCustomers,
  getCustomerById,
};