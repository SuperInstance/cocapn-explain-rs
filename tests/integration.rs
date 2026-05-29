use cocapn_explain::*;
use std::collections::HashMap;

#[test]
fn test_feature_contribution_directions() {
    let pos = FeatureContribution::new("income", 50000.0, 0.4, 0.0);
    assert_eq!(pos.direction(), "positive");
    let neg = FeatureContribution::new("debt", 10000.0, -0.3, 0.0);
    assert_eq!(neg.direction(), "negative");
    let zero = FeatureContribution::new("age", 30.0, 0.0, 0.0);
    assert_eq!(zero.direction(), "neutral");
}

#[test]
fn test_explanation_top_features() {
    let mut expl = Explanation::new("approve", 0.9);
    expl.add_contribution(FeatureContribution::new("a", 1.0, 0.1, 0.0));
    expl.add_contribution(FeatureContribution::new("b", 2.0, -0.5, 0.0));
    expl.add_contribution(FeatureContribution::new("c", 3.0, 0.3, 0.0));
    expl.add_contribution(FeatureContribution::new("d", 4.0, -0.2, 0.0));
    let top = expl.top_features(2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].feature, "b"); // |0.5|
    assert_eq!(top[1].feature, "c"); // |0.3|
}

#[test]
fn test_explanation_total_contribution() {
    let mut expl = Explanation::new("test", 0.5);
    expl.add_contribution(FeatureContribution::new("x", 1.0, 2.0, 0.0));
    expl.add_contribution(FeatureContribution::new("y", 2.0, -0.5, 0.0));
    assert!((expl.total_contribution() - 1.5).abs() < 1e-10);
}

#[test]
fn test_linear_explainer() {
    let mut exp = LinearExplainer::new(1.0);
    exp.set_weight("age", 0.1);
    exp.set_weight("income", 0.001);
    let features = HashMap::from([("age".into(), 30.0), ("income".into(), 50000.0)]);
    let result = exp.explain(&features, "approve");
    assert!(!result.contributions.is_empty());
    assert!(result.confidence > 0.0 && result.confidence < 1.0);
    // age contrib = 0.1 * 30 = 3, income contrib = 0.001 * 50000 = 50, total = 1 + 3 + 50 = 54
    assert!((result.total_contribution() - 53.0).abs() < 1e-10);
}

#[test]
fn test_permutation_importance_correlation() {
    let features = vec![
        HashMap::from([("x".into(), 1.0), ("noise".into(), 100.0)]),
        HashMap::from([("x".into(), 2.0), ("noise".into(), 50.0)]),
        HashMap::from([("x".into(), 3.0), ("noise".into(), 200.0)]),
        HashMap::from([("x".into(), 4.0), ("noise".into(), 10.0)]),
    ];
    let labels = vec![1.0, 2.0, 3.0, 4.0];
    let predict = |f: &HashMap<String, f64>| f.get("x").copied().unwrap_or(0.0);
    let imp_x = permutation_importance(&features, &labels, &predict, "x", 20);
    let imp_noise = permutation_importance(&features, &labels, &predict, "noise", 20);
    assert!(imp_x > imp_noise, "x should be more important than noise");
}

#[test]
fn test_summarize_format() {
    let mut expl = Explanation::new("reject", 0.75);
    expl.add_contribution(FeatureContribution::new("score", 0.3, -0.8, 0.0));
    let summary = expl.summarize();
    assert!(summary.starts_with("reject"));
    assert!(summary.contains("75%"));
    assert!(summary.contains("score"));
}
