# cocapn-explain-rs

Rust port of [cocapn-explain](https://github.com/SuperInstance/cocapn-explain) — decision explainability and feature importance.

## Features

- **Feature contributions**: per-feature attribution with direction and magnitude
- **Explanations**: structured decision explanations with top-k features and text summaries
- **Linear explainer**: configurable weight-based explanations
- **Permutation importance**: measure feature importance by shuffling

## Usage

```rust
use cocapn_explain::{LinearExplainer, permutation_importance};
use std::collections::HashMap;

// Linear explainer
let mut explainer = LinearExplainer::new(0.0);
explainer.set_weight("income", 0.8);
explainer.set_weight("debt", -0.6);

let features = HashMap::from([
    ("income".to_string(), 50000.0),
    ("debt".to_string(), 10000.0),
]);
let explanation = explainer.explain(&features, "approve");
println!("{}", explanation.summarize());

// Permutation importance
let importance = permutation_importance(
    &test_features, &test_labels, &predict_fn, "income", 100
);
```

## License

MIT
