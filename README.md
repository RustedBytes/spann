# spann

A proof-of-concept implementation of a SPANN-style approximate nearest neighbor index in Rust.
This project focuses on a readable, hackable baseline that still exercises core SPANN ideas:
coarse centroids, soft assignment (epsilon closure), and a simple on-disk vector store.

## Highlights

- SPANN-like coarse-to-fine search with centroid pruning.
- Soft assignment via epsilon closure to reduce recall loss.
- On-disk vector storage with lazy reads during query time.
- `f16` and `f32` vector support through a shared `Scalar` trait.
- Optional SSE2/AVX2 distance kernels for `f32` on x86/x86_64.

## Quick start

Build and run the demo with the default temp-backed store:

```bash
cargo run --release
```

Persist the vector store into a directory:

```bash
cargo run --release -- /path/to/store_dir
```

Optionally pass `vectors_per_file` as a second argument to control batch size:

```bash
cargo run --release -- /path/to/store_dir 2048
```

The demo in `src/main.rs` generates 1,000,000 `f16` vectors of dimension 128 (two clusters),
builds the index with `k=2` centroids and `epsilon=0.15`, and runs a `k=3` query against it.

## Library usage

Create an index and query it from your own code:

```rust
use ndarray::Array1;
use spann::SpannIndex;

let dimension = 128;
let data: Vec<Array1<f32>> = /* your vectors */;

let k_centroids = 64;
let epsilon_closure = 0.2;
let index = SpannIndex::build(dimension, data, k_centroids, epsilon_closure)?;

let query: Array1<f32> = /* query vector */;
let k = 10;
let rng_factor = 0.1;
let results = index.search(&query, k, rng_factor);
```

`results` contains `(vector_id, squared_distance)` pairs ordered from nearest to farthest.

## How it works (at a high level)

1. **Build**
   - Seed `k` centroids from the first vectors and run a small, fixed number of k-means
     iterations for refinement.
   - Assign each vector to all centroids within an epsilon-closure distance.
   - Store raw vectors on disk in batch files inside a directory.

2. **Search**
   - Compute distances from the query to all centroids.
   - Keep only centroids within `(1 + rng_factor)` of the nearest.
   - Scan candidate posting lists, read vectors from disk, and maintain a top-k heap.

## Project layout

- `src/lib.rs` - SPANN index, data store, and distance kernels.
- `src/main.rs` - CLI demo that builds and queries the index.
- `Cargo.toml` - crate metadata and dependencies.

## Notes and limitations

- This is a POC, not a production index. There is no delete path, and centroids are not updated.
- The data store uses batch files; vector IDs map to positions within a batch, and new vectors can be appended.
- The demo dataset is intentionally large and can be slow or memory-heavy on small machines.
- Distance is squared Euclidean; there is no cosine or inner-product mode.

## License

UNLICENSED.
