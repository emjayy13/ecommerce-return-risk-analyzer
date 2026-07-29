const Customer = require("../models/Customer");
const Return = require("../models/Return");
const getSummary = async (req, res) => {
  try {
    // Total returns across all customers
    const customers = await Customer.find();

    const totalReturns = customers.reduce(
      (sum, customer) => sum + customer.totalReturns,
      0
    );

    const totalReturnRatio = customers.reduce((sum, customer) => {
      if (customer.totalOrders === 0) return sum;

      return sum + customer.totalReturns / customer.totalOrders;
    }, 0);

    const avgReturnRatio =
      customers.length > 0
        ? Number((totalReturnRatio / customers.length).toFixed(2))
        : 0;

    const highRiskCustomers = customers.filter(
      (customer) => customer.riskLevel === "High"
    ).length;

    res.status(200).json({
      success: true,
      data: {
        totalReturns,
        avgReturnRatio,
        highRiskCustomers,
      },
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

const getCategoryStats = async (req, res) => {
  try {
    const categoryStats = await Return.aggregate([
      {
        $group: {
          _id: "$category",
          count: { $sum: 1 },
        },
      },
      {
        $project: {
          _id: 0,
          category: "$_id",
          count: 1,
        },
      },
      {
        $sort: {
          count: -1,
        },
      },
    ]);

    res.status(200).json({
      success: true,
      data: categoryStats,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

const getReturnTrends = async (req, res) => {
  try {
    const trends = await Return.aggregate([
      {
        $group: {
          _id: {
            $dateToString: {
              format: "%Y-%m-%d",
              date: "$returnedAt",
            },
          },
          returns: {
            $sum: 1,
          },
        },
      },
      {
        $project: {
          _id: 0,
          date: "$_id",
          returns: 1,
        },
      },
      {
        $sort: {
          date: 1,
        },
      },
    ]);

    res.status(200).json({
      success: true,
      data: trends,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
};

module.exports = {
  getSummary,
  getCategoryStats,
  getReturnTrends,
};