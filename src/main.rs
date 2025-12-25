use half::f16;
use ndarray::Array1;

fn main() {
    let dimension = 128;
    // Larger demo data set: two clusters of 3D vectors.
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

    let metadata: Vec<usize> = (0..data.len()).collect();
    let mut data = Some(data);
    let mut metadata = Some(metadata);

    let k_centroids = 2;
    let epsilon_closure = 0.15;

    let mut args = std::env::args().skip(1);
    let store_dir = args.next();
    let vectors_per_file = match args.next() {
        Some(value) => match value.parse::<usize>() {
            Ok(v) => Some(v),
            Err(_) => {
                eprintln!("Invalid vectors_per_file value: {value}");
                return;
            }
        },
        None => None,
    };

    let index = match (store_dir, vectors_per_file) {
        (Some(dir), Some(vectors_per_file)) => {
            spann::SpannIndex::<f16, usize>::build_with_metadata_and_store_dir_and_batch(
                dimension,
                data.take().expect("Data already moved"),
                metadata.take().expect("Metadata already moved"),
                k_centroids,
                epsilon_closure,
                dir,
                vectors_per_file,
            )
        }
        (Some(dir), None) => spann::SpannIndex::<f16, usize>::build_with_metadata_and_store_dir(
            dimension,
            data.take().expect("Data already moved"),
            metadata.take().expect("Metadata already moved"),
            k_centroids,
            epsilon_closure,
            dir,
        ),
        (None, Some(vectors_per_file)) => {
            spann::SpannIndex::<f16, usize>::build_with_metadata_and_vectors_per_file(
                dimension,
                data.take().expect("Data already moved"),
                metadata.take().expect("Metadata already moved"),
                k_centroids,
                epsilon_closure,
                vectors_per_file,
            )
        }
        (None, None) => spann::SpannIndex::<f16, usize>::build_with_metadata(
            dimension,
            data.take().expect("Data already moved"),
            metadata.take().expect("Metadata already moved"),
            k_centroids,
            epsilon_closure,
        ),
    };
    let index = match index {
        Ok(index) => index,
        Err(err) => {
            eprintln!("Failed to build index: {err}");
            return;
        }
    };

    let mut query_vec = Vec::with_capacity(dimension);
    query_vec.push(f16::from_f32(1.0));
    query_vec.push(f16::from_f32(1.0));
    query_vec.push(f16::from_f32(1.2));
    for j in 3..dimension {
        let j_f32 = j as f32;
        query_vec.push(f16::from_f32(1.0 + j_f32 * 0.001));
    }
    let query = Array1::from(query_vec);
    let k = 3;
    let rng_factor = 0.1;

    let results = index.search_with_metadata(&query, k, rng_factor);

    println!("Query: {:?}", query);
    println!("Top {k} results (index, squared_distance, metadata):");
    for (id, dist, meta) in results {
        println!("  {id}: {dist:.4} (meta {meta})");
    }
}
