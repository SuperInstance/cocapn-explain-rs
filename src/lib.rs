//! Decision explainability — feature importance, SHAP-like attribution.

use std::collections::HashMap;

/// A feature contribution to a decision.
#[derive(Debug, Clone)]
pub struct FeatureContribution {
    pub feature: String,
    pub value: f64,
    pub contribution: f64,
    pub baseline: f64,
}

impl FeatureContribution {
    pub fn new(feature: &str, value: f64, contribution: f64, baseline: f64) -> Self {
        Self { feature: feature.to_string(), value, contribution, baseline }
    }
    pub fn direction(&self) -> &'static str {
        if self.contribution > 0.0 { "positive" } else if self.contribution < 0.0 { "negative" } else { "neutral" }
    }
}

/// An explanation of a decision.
#[derive(Debug, Clone)]
pub struct Explanation {
    pub decision: String,
    pub confidence: f64,
    pub contributions: Vec<FeatureContribution>,
    pub summary: String,
}

impl Explanation {
    pub fn new(decision: &str, confidence: f64) -> Self {
        Self { decision: decision.to_string(), confidence, contributions: vec![], summary: String::new() }
    }

    pub fn add_contribution(&mut self, c: FeatureContribution) { self.contributions.push(c); }

    /// Top-k features by absolute contribution.
    pub fn top_features(&self, k: usize) -> Vec<&FeatureContribution> {
        let mut sorted: Vec<_> = self.contributions.iter().collect();
        sorted.sort_by(|a, b| b.contribution.abs().partial_cmp(&a.contribution.abs()).unwrap());
        sorted.into_iter().take(k).collect()
    }

    /// Sum of all contributions.
    pub fn total_contribution(&self) -> f64 {
        self.contributions.iter().map(|c| c.contribution).sum()
    }

    /// Generate a text summary.
    pub fn summarize(&self) -> String {
        let top = self.top_features(3);
        let parts: Vec<String> = top.iter().map(|c| {
            format!("{}({:.2},{:+.3})", c.feature, c.value, c.contribution)
        }).collect();
        format!("{} [{:.0}%] ← {}", self.decision, self.confidence * 100.0, parts.join(", "))
    }
}

/// Simple linear feature importance explainer.
#[derive(Debug, Clone)]
pub struct LinearExplainer {
    pub weights: HashMap<String, f64>,
    pub baseline: f64,
}

impl LinearExplainer {
    pub fn new(baseline: f64) -> Self { Self { weights: HashMap::new(), baseline } }
    pub fn set_weight(&mut self, feature: &str, weight: f64) { self.weights.insert(feature.to_string(), weight); }

    pub fn explain(&self, features: &HashMap<String, f64>, decision: &str) -> Explanation {
        let mut expl = Explanation::new(decision, 0.0);
        let mut total = self.baseline;
        for (feat, &value) in features {
            let weight = self.weights.get(feat).copied().unwrap_or(0.0);
            let contrib = weight * value;
            expl.add_contribution(FeatureContribution::new(feat, value, contrib, 0.0));
            total += contrib;
        }
        // Sigmoid for confidence
        expl.confidence = 1.0 / (1.0 + (-total).exp());
        expl.summary = expl.summarize();
        expl
    }
}

/// Permutation importance: measure how much shuffling a feature hurts performance.
pub fn permutation_importance(
    features: &[HashMap<String, f64>],
    labels: &[f64],
    predict: &dyn Fn(&HashMap<String, f64>) -> f64,
    feature_name: &str,
    n_permutations: usize,
) -> f64 {
    // Baseline score
    let baseline_mse: f64 = features.iter().zip(labels.iter())
        .map(|(f, l)| (predict(f) - l).powi(2)).sum::<f64>() / features.len() as f64;

    let mut permuted_mse = 0.0;
    let mut rng_state = 42u64;
    let mut rng = || { rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1); rng_state };

    for _ in 0..n_permutations {
        let mut permuted: Vec<HashMap<String, f64>> = features.to_vec();
        // Fisher-Yates shuffle the feature values
        let n = permuted.len();
        for i in (1..n).rev() {
            let j = (rng() as usize) % (i + 1);
            let vi = permuted[i].get(feature_name).copied().unwrap_or(0.0);
            let vj = permuted[j].get(feature_name).copied().unwrap_or(0.0);
            permuted[i].insert(feature_name.to_string(), vj);
            permuted[j].insert(feature_name.to_string(), vi);
        }
        let mse: f64 = permuted.iter().zip(labels.iter())
            .map(|(f, l)| (predict(f) - l).powi(2)).sum::<f64>() / n as f64;
        permuted_mse += mse;
    }
    permuted_mse /= n_permutations as f64;
    permuted_mse - baseline_mse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_contribution() {
        let c = FeatureContribution::new("age", 25.0, 0.3, 0.0);
        assert_eq!(c.direction(), "positive");
        let c2 = FeatureContribution::new("risk", 0.9, -0.5, 0.0);
        assert_eq!(c2.direction(), "negative");
    }

    #[test]
    fn test_explanation() {
        let mut expl = Explanation::new("approve", 0.85);
        expl.add_contribution(FeatureContribution::new("income", 50000.0, 0.4, 0.0));
        expl.add_contribution(FeatureContribution::new("debt", 10000.0, -0.2, 0.0));
        assert_eq!(expl.contributions.len(), 2);
        assert!((expl.total_contribution() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_top_features() {
        let mut expl = Explanation::new("test", 0.5);
        expl.add_contribution(FeatureContribution::new("a", 1.0, 0.1, 0.0));
        expl.add_contribution(FeatureContribution::new("b", 2.0, -0.5, 0.0));
        expl.add_contribution(FeatureContribution::new("c", 3.0, 0.3, 0.0));
        let top = expl.top_features(2);
        assert_eq!(top[0].feature, "b");
        assert_eq!(top[1].feature, "c");
    }

    #[test]
    fn test_linear_explainer() {
        let mut explainer = LinearExplainer::new(0.0);
        explainer.set_weight("x", 2.0);
        explainer.set_weight("y", -1.0);
        let features = HashMap::from([("x".to_string(), 3.0), ("y".to_string(), 1.0)]);
        let expl = explainer.explain(&features, "predict");
        assert!((expl.total_contribution() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_summarize() {
        let mut expl = Explanation::new("approve", 0.9);
        expl.add_contribution(FeatureContribution::new("score", 0.8, 0.5, 0.0));
        let summary = expl.summarize();
        assert!(summary.contains("approve"));
    }

    #[test]
    fn test_permutation_importance() {
        let features = vec![
            HashMap::from([("x".to_string(), 1.0), ("y".to_string(), 0.0)]),
            HashMap::from([("x".to_string(), 2.0), ("y".to_string(), 1.0)]),
            HashMap::from([("x".to_string(), 3.0), ("y".to_string(), 0.0)]),
        ];
        let labels = vec![1.0, 2.0, 3.0];
        let predict = |f: &HashMap<String, f64>| f.get("x").copied().unwrap_or(0.0);
        let imp_x = permutation_importance(&features, &labels, &predict, "x", 10);
        let imp_y = permutation_importance(&features, &labels, &predict, "y", 10);
        // x should be more important since the predictor uses it
        assert!(imp_x > imp_y);
    }
}
