use ndarray::Array1;
use half::f16;

fn main() {
    // Larger demo data set: two clusters of 3D vectors.
    let mut data: Vec<Array1<f16>> = Vec::new();
    for i in 0..2_000_000 {
        let i_f32 = i as f32;
        let t = i_f32 * 0.01;
        let mod_component = (i % 10) as f32 * 0.02;
        data.push(Array1::from(vec![
            f16::from_f32(1.0 + t),
            f16::from_f32(1.0 - t * 0.5),
            f16::from_f32(1.0 + mod_component),
        ]));
    }
    for i in 0..2_000_000 {
        let i_f32 = i as f32;
        let t = i_f32 * 0.01;
        let mod_component = (i % 12) as f32 * 0.015;
        data.push(Array1::from(vec![
            f16::from_f32(5.0 - t * 0.4),
            f16::from_f32(5.0 + t * 0.3),
            f16::from_f32(5.0 - mod_component),
        ]));
    }

    let dimension = 3;
    let k_centroids = 2;
    let epsilon_closure = 0.15;

    let store_dir = std::env::args().nth(1);
    let index = match store_dir {
        Some(dir) => spann::SpannIndex::<f16>::build_with_store_dir(
            dimension,
            data,
            k_centroids,
            epsilon_closure,
            dir,
        ),
        None => spann::SpannIndex::<f16>::build(dimension, data, k_centroids, epsilon_closure),
    };
    let index = match index {
        Ok(index) => index,
        Err(err) => {
            eprintln!("Failed to build index: {err}");
            return;
        }
    };

    let query = Array1::from(vec![
        f16::from_f32(1.0),
        f16::from_f32(1.0),
        f16::from_f32(1.2),
    ]);
    let k = 3;
    let rng_factor = 0.1;

    let results = index.search(&query, k, rng_factor);

    println!("Query: {:?}", query);
    println!("Top {k} results (index, squared_distance):");
    for (id, dist) in results {
        println!("  {id}: {dist:.4}");
    }
}
