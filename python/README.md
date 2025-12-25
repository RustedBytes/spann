# spann (Python)

Python bindings for the `spann` Rust crate, built with PyO3.

## Build

```bash
maturin develop
```

## Usage

```python
import spann

index = spann.SpannIndexF32.build(
    vectors=[[0.1, 0.2, 0.3], [0.9, 1.0, 1.1]],
    k_centroids=1,
    epsilon_closure=0.2,
)

results = index.search([0.1, 0.2, 0.3], k=1, rng_factor=0.1)
print(results)
```
