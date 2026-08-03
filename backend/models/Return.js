const mongoose = require("mongoose");

const returnSchema = new mongoose.Schema(
  {
    customer: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Customer",
      required: true,
    },

    orderId: {
      type: String,
      required: true,
      unique: true,
      trim: true,
    },

    category: {
      type: String,
      required: true,
      enum: [
        "Electronics",
        "Fashion",
        "Shoes",
        "Home",
        "Beauty",
        "Sports",
        "Others",
      ],
    },

    reason: {
      type: String,
      required: true,
      trim: true,
    },

    daysToReturn: {
      type: Number,
      required: true,
      min: 0,
    },

    customerRating: {
      type: Number,
      required: true,
      min: 1,
      max: 5,
    },

    mismatchFlag: {
      type: Boolean,
      default: false,
    },

    returnedAt: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Return", returnSchema);