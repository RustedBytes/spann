use clap::{Parser, Subcommand};
use half::f16;
use ndarray::Array1;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "spann-demo", about = "SPANN proof-of-concept CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Create {
        /// Path to write the index file.
        index_path: PathBuf,
        /// Directory where vector batches are stored.
        #[arg(long)]
        store_dir: Option<PathBuf>,
        /// Override batch size for on-disk vector storage.
        #[arg(long)]
        vectors_per_file: Option<usize>,
    },
    Query {
        /// Path to an existing index file.
        index_path: PathBuf,
        /// Number of neighbors to return.
        #[arg(long, default_value_t = 3)]
        k: usize,
        /// Pruning factor for centroid selection.
        #[arg(long, default_value_t = 0.1)]
        rng_factor: f32,
    },
}

const DIMENSION: usize = 128;
const K_CENTROIDS: usize = 2;
const EPSILON_CLOSURE: f32 = 0.15;

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Create {
            index_path,
            store_dir,
            vectors_per_file,
        } => {
            let store_dir = store_dir.unwrap_or_else(|| {
                index_path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join("store")
            });
            if let Err(err) = create_index(index_path, store_dir, vectors_per_file) {
                eprintln!("{err}");
            }
        }
        Commands::Query {
            index_path,
            k,
            rng_factor,
        } => {
            if let Err(err) = query_index(index_path, k, rng_factor) {
                eprintln!("{err}");
            }
        }
    }
}

fn create_index(
    index_path: PathBuf,
    store_dir: PathBuf,
    vectors_per_file: Option<usize>,
) -> Result<(), String> {
    let data = build_demo_vectors(DIMENSION);
    let metadata: Vec<usize> = (0..data.len()).collect();

    let index = match vectors_per_file {
        Some(vectors_per_file) => {
            spann::SpannIndex::<f16, usize>::build_with_metadata_and_store_dir_and_batch(
                DIMENSION,
                data,
                metadata,
                K_CENTROIDS,
                EPSILON_CLOSURE,
                &store_dir,
                vectors_per_file,
            )
        }
        None => spann::SpannIndex::<f16, usize>::build_with_metadata_and_store_dir(
            DIMENSION,
            data,
            metadata,
            K_CENTROIDS,
            EPSILON_CLOSURE,
            &store_dir,
        ),
    }
    .map_err(|err| format!("Failed to build index: {err}"))?;

    index
        .save_to_path(&index_path)
        .map_err(|err| format!("Failed to save index: {err}"))?;

    println!(
        "Index saved to {} (vector store at {}).",
        index_path.display(),
        store_dir.display()
    );
    Ok(())
}

fn query_index(index_path: PathBuf, k: usize, rng_factor: f32) -> Result<(), String> {
    let index = spann::SpannIndex::<f16, usize>::load_from_path(&index_path)
        .map_err(|err| format!("Failed to load index {}: {err}", index_path.display()))?;

    let query = build_query_vector(index.dimension());
    let results = index.search_with_metadata(&query, k, rng_factor);

    println!("Query: {:?}", query);
    println!("Top {k} results (index, squared_distance, metadata):");
    for (id, dist, meta) in results {
        println!("  {id}: {dist:.4} (meta {meta})");
    }

    Ok(())
}

fn build_demo_vectors(dimension: usize) -> Vec<Array1<f16>> {
    let mut data: Vec<Array1<f16>> = Vec::new();
    for i in 0..500_000 {
        let i_f32 = i as f32;
        let t = i_f32 * 0.01;
        let mod_component = (i % 10) as f32 * 0.02;
        let mut vec = Vec::with_capacity(dimension);
        vec.push(f16::from_f32(1.0 + t));
        vec.push(f16::from_f32(1.0 - t * 0.5));
        vec.push(f16::from_f32(1.0 + mod_component));
        for j in 3..dimension {
            let j_f32 = j as f32;
            vec.push(f16::from_f32(1.0 + mod_component + j_f32 * 0.001));
        }
        data.push(Array1::from(vec));
    }
    for i in 0..500_000 {
        let i_f32 = i as f32;
        let t = i_f32 * 0.01;
        let mod_component = (i % 12) as f32 * 0.015;
        let mut vec = Vec::with_capacity(dimension);
        vec.push(f16::from_f32(5.0 - t * 0.4));
        vec.push(f16::from_f32(5.0 + t * 0.3));
        vec.push(f16::from_f32(5.0 - mod_component));
        for j in 3..dimension {
            let j_f32 = j as f32;
            vec.push(f16::from_f32(5.0 - mod_component - j_f32 * 0.001));
        }
        data.push(Array1::from(vec));
    }
    data
}

fn build_query_vector(dimension: usize) -> Array1<f16> {
    let mut query_vec = Vec::with_capacity(dimension);
    query_vec.push(f16::from_f32(1.0));
    query_vec.push(f16::from_f32(1.0));
    query_vec.push(f16::from_f32(1.2));
    for j in 3..dimension {
        let j_f32 = j as f32;
        query_vec.push(f16::from_f32(1.0 + j_f32 * 0.001));
    }
    Array1::from(query_vec)
}
