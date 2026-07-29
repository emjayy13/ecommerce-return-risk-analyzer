const mongoose = require("mongoose");

const customerSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
      trim: true,
    },

    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
    },

    totalOrders: {
      type: Number,
      required: true,
      default: 0,
      min: 0,
    },

    totalReturns: {
      type: Number,
      required: true,
      default: 0,
      min: 0,
    },

    flaggedBefore: {
      type: Boolean,
      default: false,
    },

    riskScore: {
      type: Number,
      default: 0,
      min: 0,
      max: 100,
    },

    riskLevel: {
      type: String,
      enum: ["Low", "Medium", "High"],
      default: "Low",
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Customer", customerSchema);