# spann (Python)

Python bindings for the `spann` Rust crate, built with PyO3.

## Build

```bash
maturin build --release --out dist
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

## Metadata + on-disk store

```python
import spann
from pathlib import Path

base_dir = Path("my-index")
index = spann.SpannIndexF32Meta.build_with_metadata_in_dir(
    vectors=[[0.1, 0.2, 0.3], [0.9, 1.0, 1.1]],
    metadata=[100, 200],
    k_centroids=1,
    epsilon_closure=0.2,
    base_dir=base_dir,
)

loaded = spann.SpannIndexF32Meta.load_from_dir(base_dir)
results = loaded.search_with_metadata([0.1, 0.2, 0.3], k=1, rng_factor=0.1)
print(results)
```
