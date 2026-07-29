const RISKY_CATEGORIES = ["Fashion", "Shoes"];

const calculateRiskScore = ({
  returnRatio,
  avgReturnWindow,
  riskyCategoryCount,
  vagueReasonCount,
  avgCustomerRating,
  mismatchHistory,
}) => {
  let score = 0;

  // Rule 1
  if (returnRatio > 0.4) {
    score += 30;
  }

  // Rule 2
  if (avgReturnWindow < 2) {
    score += 15;
  }

  // Rule 3
  if (riskyCategoryCount > 0) {
    score += 20;
  }

  // Rule 4
  if (vagueReasonCount > 2) {
    score += 25;
  }

  // Low ratings + frequent returns
  if (avgCustomerRating <= 2.5) {
    score += 10;
  }

  // Previous mismatch history
  if (mismatchHistory > 0) {
    score += 20;
  }

  // Keep score within 0–100
  score = Math.min(score, 100);

  let riskLevel = "Low";

  if (score > 70) {
    riskLevel = "High";
  } else if (score > 40) {
    riskLevel = "Medium";
  }

  return {
    score,
    riskLevel,
  };
}

module.exports = {
  calculateRiskScore,
  RISKY_CATEGORIES,
};



