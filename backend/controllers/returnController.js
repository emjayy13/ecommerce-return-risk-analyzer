const Customer = require("../models/Customer");
const Return = require("../models/Return");
const { predictRisk } = require("../services/mlService");
const {
  calculateRiskScore,
  RISKY_CATEGORIES,
} = require("../services/riskCalculator");
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

    existingCustomer.totalReturns += 1;

    const customerReturns = await Return.find({
      customer: existingCustomer._id,
    });

    const returnRatio =
      existingCustomer.totalReturns / existingCustomer.totalOrders;

    const avgReturnWindow =
      customerReturns.reduce(
        (sum, item) => sum + item.daysToReturn,
        0
      ) / customerReturns.length;

    const riskyCategoryCount = customerReturns.filter((item) =>
      RISKY_CATEGORIES.includes(item.category)
    ).length;

    const vagueReasonCount = customerReturns.filter((item) =>
      ["Defective", "Other"].includes(item.reason)
    ).length;

    const avgCustomerRating =
    customerReturns.reduce(
      (sum, item) => sum + item.customerRating,
      0
    ) / customerReturns.length;

    const mismatchHistory = customerReturns.filter(
      (item) => item.mismatchFlag
    ).length;

    // const { score, riskLevel } = calculateRiskScore({
    //   returnRatio,
    //   avgReturnWindow,
    //   riskyCategoryCount,
    //   vagueReasonCount,
    //   avgCustomerRating,
    //   mismatchHistory,
    // });

    // existingCustomer.riskScore = score;
    // existingCustomer.riskLevel = riskLevel;

      const prediction = await predictRisk({
      customer_id: existingCustomer._id.toString(),
      order_id: orderId,
      product_category: category,
      return_reason: reason,
      is_returned: true,
    });

    //console.log("ML Prediction:", prediction);

    const score = prediction.risk_score;
    const riskLevel = prediction.risk_level;
    
    
    await existingCustomer.save();


    res.status(201).json({
      success: true,
      message: "Return created successfully",
      data: {
        return: newReturn,
        riskScore: score,
        riskLevel: riskLevel,
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
  createReturn,
};