# cocapn-explain-rs

Decision explainability and feature importance — per-feature attribution with direction and magnitude, SHAP-like linear explanations, and permutation importance scoring.

## What This Gives You

- **Feature contributions** — Per-feature attribution with value, contribution, baseline, and direction (positive/negative/neutral)
- **`LinearExplainer`** — Configurable weight-based explanations for linear models
- **Permutation importance** — Measure feature importance by shuffling columns and measuring accuracy drop
- **Structured explanations** — Top-k features, total contribution, text summaries
- **Zero external dependencies** — Pure Rust, `std` + `HashMap` only

## Quick Start

```rust
use cocapn_explain::{LinearExplainer, permutation_importance};
use std::collections::HashMap;

// Create a linear explainer with weights
let mut explainer = LinearExplainer::new(0.0);
explainer.set_weight("income", 0.8);
explainer.set_weight("debt", -0.6);

let features = HashMap::from([
    ("income".to_string(), 50000.0),
    ("debt".to_string(), 10000.0),
]);

let explanation = explainer.explain(&features, "approve");
println!("{}", explanation.summarize());
// Top features by absolute contribution

// Permutation importance: how much does shuffling "income" hurt predictions?
let importance = permutation_importance(
    &test_features, &test_labels, &predict_fn, "income", 100
);
```

## API Reference

### `FeatureContribution`

| Field | Description |
|-------|-------------|
| `feature` | Feature name |
| `value` | Actual feature value |
| `contribution` | How much this feature pushed the decision |
| `baseline` | Reference value |
| `direction()` | `"positive"`, `"negative"`, or `"neutral"` |

### `Explanation`

| Method | Description |
|--------|-------------|
| `new(decision, confidence)` | Create explanation for a decision |
| `add_contribution(c)` | Add a feature contribution |
| `top_features(k)` | Top-k features by absolute contribution |
| `total_contribution()` | Sum of all contributions |
| `summarize()` | Human-readable text summary |

### `LinearExplainer`

```rust
LinearExplainer::new(baseline)    // Create with baseline value
explainer.set_weight(feature, w)  // Set feature weight
explainer.explain(&features, decision) -> Explanation
```

### `permutation_importance`

```rust
permutation_importance(features, labels, predict_fn, feature_name, n_shuffles) -> f64
```

## How It Fits

- **[causal-graph-rs](https://github.com/SuperInstance/causal-graph-rs)** — Causal structure informs which features are truly causal vs confounded
- **[cocapn-health-rs](https://github.com/SuperInstance/cocapn-health-rs)** — Explain why a service was flagged as unhealthy
- **[commit-predictor](https://github.com/SuperInstance/commit-predictor)** — Explain which features drive commit activity predictions
- **[constraint-dsl](https://github.com/SuperInstance/constraint-dsl)** — Explain which constraints had the most impact on a pipeline result

## Testing

6 tests covering feature contributions, linear explanations, permutation importance, and text summaries.

```bash
cargo test
```

## Installation

```toml
[dependencies]
cocapn-explain = { git = "https://github.com/SuperInstance/cocapn-explain-rs" }
```

```bash
git clone https://github.com/SuperInstance/cocapn-explain-rs.git
cd cocapn-explain-rs
cargo build
```

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance) ecosystem.
