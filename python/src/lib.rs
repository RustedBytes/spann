use std::path::{Path, PathBuf};

use ::spann::SpannIndex;
use half::f16;
use ndarray::Array1;
use pyo3::Bound;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

const CARGO_PKG_VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_INDEX_FILENAME: &str = "spann.index";

fn runtime_err(message: String) -> PyErr {
    PyRuntimeError::new_err(message)
}

fn ensure_vectors_f32(vectors: &[Vec<f32>]) -> PyResult<usize> {
    if vectors.is_empty() {
        return Err(PyValueError::new_err("vectors must be non-empty"));
    }
    let dimension = vectors[0].len();
    if dimension == 0 {
        return Err(PyValueError::new_err(
            "vectors must contain at least one element",
        ));
    }
    for (idx, vector) in vectors.iter().enumerate().skip(1) {
        if vector.len() != dimension {
            return Err(PyValueError::new_err(format!(
                "vector {idx} has dimension {}, expected {}",
                vector.len(),
                dimension
            )));
        }
    }
    Ok(dimension)
}

fn ensure_k_centroids(k_centroids: usize) -> PyResult<()> {
    if k_centroids == 0 {
        return Err(PyValueError::new_err("k_centroids must be >= 1"));
    }
    Ok(())
}

fn vectors_to_array1_f32(vectors: Vec<Vec<f32>>, dimension: usize) -> PyResult<Vec<Array1<f32>>> {
    let mut output = Vec::with_capacity(vectors.len());
    for (idx, vector) in vectors.into_iter().enumerate() {
        if vector.len() != dimension {
            return Err(PyValueError::new_err(format!(
                "vector {idx} has dimension {}, expected {}",
                vector.len(),
                dimension
            )));
        }
        output.push(Array1::from(vector));
    }
    Ok(output)
}

fn vectors_to_array1_f16(vectors: Vec<Vec<f32>>, dimension: usize) -> PyResult<Vec<Array1<f16>>> {
    let mut output = Vec::with_capacity(vectors.len());
    for (idx, vector) in vectors.into_iter().enumerate() {
        if vector.len() != dimension {
            return Err(PyValueError::new_err(format!(
                "vector {idx} has dimension {}, expected {}",
                vector.len(),
                dimension
            )));
        }
        let mut converted = Vec::with_capacity(vector.len());
        for value in vector {
            converted.push(f16::from_f32(value));
        }
        output.push(Array1::from(converted));
    }
    Ok(output)
}

fn vector_to_array1_f32(vector: Vec<f32>, dimension: usize, label: &str) -> PyResult<Array1<f32>> {
    if vector.len() != dimension {
        return Err(PyValueError::new_err(format!(
            "{label} has dimension {}, expected {}",
            vector.len(),
            dimension
        )));
    }
    Ok(Array1::from(vector))
}

fn vector_to_array1_f16(vector: Vec<f32>, dimension: usize, label: &str) -> PyResult<Array1<f16>> {
    if vector.len() != dimension {
        return Err(PyValueError::new_err(format!(
            "{label} has dimension {}, expected {}",
            vector.len(),
            dimension
        )));
    }
    let mut converted = Vec::with_capacity(vector.len());
    for value in vector {
        converted.push(f16::from_f32(value));
    }
    Ok(Array1::from(converted))
}

fn index_path_from_dir(dir: &Path) -> PathBuf {
    dir.join(DEFAULT_INDEX_FILENAME)
}

#[pyfunction]
fn cargo_version() -> &'static str {
    CARGO_PKG_VERSION
}

#[pyclass]
struct SpannIndexF32 {
    inner: SpannIndex<f32, ()>,
}

#[pymethods]
impl SpannIndexF32 {
    #[staticmethod]
    fn build(vectors: Vec<Vec<f32>>, k_centroids: usize, epsilon_closure: f32) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        let data = vectors_to_array1_f32(vectors, dimension)?;
        let inner = SpannIndex::build(dimension, data, k_centroids, epsilon_closure)
            .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_vectors_per_file(
        vectors: Vec<Vec<f32>>,
        k_centroids: usize,
        epsilon_closure: f32,
        vectors_per_file: usize,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        let data = vectors_to_array1_f32(vectors, dimension)?;
        let inner = SpannIndex::build_with_vectors_per_file(
            dimension,
            data,
            k_centroids,
            epsilon_closure,
            vectors_per_file,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_store_dir(
        vectors: Vec<Vec<f32>>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: PathBuf,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        let data = vectors_to_array1_f32(vectors, dimension)?;
        let inner = SpannIndex::build_with_store_dir(
            dimension,
            data,
            k_centroids,
            epsilon_closure,
            store_dir,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_store_dir_and_batch(
        vectors: Vec<Vec<f32>>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: PathBuf,
        vectors_per_file: usize,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        let data = vectors_to_array1_f32(vectors, dimension)?;
        let inner = SpannIndex::build_with_store_dir_and_batch(
            dimension,
            data,
            k_centroids,
            epsilon_closure,
            store_dir,
            vectors_per_file,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn epsilon_closure(&self) -> f32 {
        self.inner.epsilon_closure()
    }

    fn add_vector(&mut self, vector: Vec<f32>) -> PyResult<usize> {
        let array = vector_to_array1_f32(vector, self.inner.dimension(), "vector")?;
        self.inner.add_vector(array).map_err(runtime_err)
    }

    fn search(&self, query: Vec<f32>, k: usize, rng_factor: f32) -> PyResult<Vec<(usize, f32)>> {
        let array = vector_to_array1_f32(query, self.inner.dimension(), "query")?;
        Ok(self.inner.search(&array, k, rng_factor))
    }
}

#[pyclass]
struct SpannIndexF32Meta {
    inner: SpannIndex<f32, usize>,
}

#[pymethods]
impl SpannIndexF32Meta {
    #[staticmethod]
    fn build_with_metadata_in_dir(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
        base_dir: PathBuf,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f32(vectors, dimension)?;
        let store_dir = base_dir.clone();
        let inner = SpannIndex::build_with_metadata_and_store_dir(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
            store_dir,
        )
        .map_err(runtime_err)?;
        let index = Self { inner };
        index.save_to_dir(base_dir)?;
        Ok(index)
    }

    #[staticmethod]
    fn build_with_metadata_in_dir_and_batch(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
        base_dir: PathBuf,
        vectors_per_file: usize,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f32(vectors, dimension)?;
        let store_dir = base_dir.clone();
        let inner = SpannIndex::build_with_metadata_and_store_dir_and_batch(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
            store_dir,
            vectors_per_file,
        )
        .map_err(runtime_err)?;
        let index = Self { inner };
        index.save_to_dir(base_dir)?;
        Ok(index)
    }

    #[staticmethod]
    fn build_with_metadata(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f32(vectors, dimension)?;
        let inner = SpannIndex::build_with_metadata(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_metadata_and_vectors_per_file(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
        vectors_per_file: usize,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f32(vectors, dimension)?;
        let inner = SpannIndex::build_with_metadata_and_vectors_per_file(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
            vectors_per_file,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_metadata_and_store_dir(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: PathBuf,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f32(vectors, dimension)?;
        let inner = SpannIndex::build_with_metadata_and_store_dir(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
            store_dir,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_metadata_and_store_dir_and_batch(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: PathBuf,
        vectors_per_file: usize,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f32(vectors, dimension)?;
        let inner = SpannIndex::build_with_metadata_and_store_dir_and_batch(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
            store_dir,
            vectors_per_file,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn load_from_dir(base_dir: PathBuf) -> PyResult<Self> {
        let index_path = index_path_from_dir(&base_dir);
        let inner = SpannIndex::<f32, usize>::load_from_path(index_path).map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = SpannIndex::<f32, usize>::load_from_path(path).map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn epsilon_closure(&self) -> f32 {
        self.inner.epsilon_closure()
    }

    fn metadata(&self, id: usize) -> Option<usize> {
        self.inner.metadata(id).copied()
    }

    fn add_vector(&mut self, vector: Vec<f32>, metadata: usize) -> PyResult<usize> {
        let array = vector_to_array1_f32(vector, self.inner.dimension(), "vector")?;
        self.inner
            .add_vector_with_metadata(array, metadata)
            .map_err(runtime_err)
    }

    fn search(&self, query: Vec<f32>, k: usize, rng_factor: f32) -> PyResult<Vec<(usize, f32)>> {
        let array = vector_to_array1_f32(query, self.inner.dimension(), "query")?;
        Ok(self.inner.search(&array, k, rng_factor))
    }

    fn search_with_metadata(
        &self,
        query: Vec<f32>,
        k: usize,
        rng_factor: f32,
    ) -> PyResult<Vec<(usize, f32, usize)>> {
        let array = vector_to_array1_f32(query, self.inner.dimension(), "query")?;
        Ok(self
            .inner
            .search_with_metadata(&array, k, rng_factor)
            .into_iter()
            .map(|(id, dist, meta)| (id, dist, *meta))
            .collect())
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_to_path(path).map_err(runtime_err)
    }

    fn save_to_dir(&self, base_dir: PathBuf) -> PyResult<()> {
        let index_path = index_path_from_dir(&base_dir);
        self.inner.save_to_path(index_path).map_err(runtime_err)
    }
}

#[pyclass]
struct SpannIndexF16 {
    inner: SpannIndex<f16, ()>,
}

#[pymethods]
impl SpannIndexF16 {
    #[staticmethod]
    fn build(vectors: Vec<Vec<f32>>, k_centroids: usize, epsilon_closure: f32) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        let data = vectors_to_array1_f16(vectors, dimension)?;
        let inner = SpannIndex::build(dimension, data, k_centroids, epsilon_closure)
            .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_vectors_per_file(
        vectors: Vec<Vec<f32>>,
        k_centroids: usize,
        epsilon_closure: f32,
        vectors_per_file: usize,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        let data = vectors_to_array1_f16(vectors, dimension)?;
        let inner = SpannIndex::build_with_vectors_per_file(
            dimension,
            data,
            k_centroids,
            epsilon_closure,
            vectors_per_file,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_store_dir(
        vectors: Vec<Vec<f32>>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: PathBuf,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        let data = vectors_to_array1_f16(vectors, dimension)?;
        let inner = SpannIndex::build_with_store_dir(
            dimension,
            data,
            k_centroids,
            epsilon_closure,
            store_dir,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_store_dir_and_batch(
        vectors: Vec<Vec<f32>>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: PathBuf,
        vectors_per_file: usize,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        let data = vectors_to_array1_f16(vectors, dimension)?;
        let inner = SpannIndex::build_with_store_dir_and_batch(
            dimension,
            data,
            k_centroids,
            epsilon_closure,
            store_dir,
            vectors_per_file,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn epsilon_closure(&self) -> f32 {
        self.inner.epsilon_closure()
    }

    fn add_vector(&mut self, vector: Vec<f32>) -> PyResult<usize> {
        let array = vector_to_array1_f16(vector, self.inner.dimension(), "vector")?;
        self.inner.add_vector(array).map_err(runtime_err)
    }

    fn search(&self, query: Vec<f32>, k: usize, rng_factor: f32) -> PyResult<Vec<(usize, f32)>> {
        let array = vector_to_array1_f16(query, self.inner.dimension(), "query")?;
        Ok(self.inner.search(&array, k, rng_factor))
    }
}

#[pyclass]
struct SpannIndexF16Meta {
    inner: SpannIndex<f16, usize>,
}

#[pymethods]
impl SpannIndexF16Meta {
    #[staticmethod]
    fn build_with_metadata_in_dir(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
        base_dir: PathBuf,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f16(vectors, dimension)?;
        let store_dir = base_dir.clone();
        let inner = SpannIndex::build_with_metadata_and_store_dir(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
            store_dir,
        )
        .map_err(runtime_err)?;
        let index = Self { inner };
        index.save_to_dir(base_dir)?;
        Ok(index)
    }

    #[staticmethod]
    fn build_with_metadata_in_dir_and_batch(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
        base_dir: PathBuf,
        vectors_per_file: usize,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f16(vectors, dimension)?;
        let store_dir = base_dir.clone();
        let inner = SpannIndex::build_with_metadata_and_store_dir_and_batch(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
            store_dir,
            vectors_per_file,
        )
        .map_err(runtime_err)?;
        let index = Self { inner };
        index.save_to_dir(base_dir)?;
        Ok(index)
    }

    #[staticmethod]
    fn build_with_metadata(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f16(vectors, dimension)?;
        let inner = SpannIndex::build_with_metadata(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_metadata_and_vectors_per_file(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
        vectors_per_file: usize,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f16(vectors, dimension)?;
        let inner = SpannIndex::build_with_metadata_and_vectors_per_file(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
            vectors_per_file,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_metadata_and_store_dir(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: PathBuf,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f16(vectors, dimension)?;
        let inner = SpannIndex::build_with_metadata_and_store_dir(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
            store_dir,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn build_with_metadata_and_store_dir_and_batch(
        vectors: Vec<Vec<f32>>,
        metadata: Vec<usize>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: PathBuf,
        vectors_per_file: usize,
    ) -> PyResult<Self> {
        ensure_k_centroids(k_centroids)?;
        let dimension = ensure_vectors_f32(&vectors)?;
        if metadata.len() != vectors.len() {
            return Err(PyValueError::new_err(format!(
                "metadata has {} entries but {} vectors were provided",
                metadata.len(),
                vectors.len()
            )));
        }
        let data = vectors_to_array1_f16(vectors, dimension)?;
        let inner = SpannIndex::build_with_metadata_and_store_dir_and_batch(
            dimension,
            data,
            metadata,
            k_centroids,
            epsilon_closure,
            store_dir,
            vectors_per_file,
        )
        .map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn load_from_dir(base_dir: PathBuf) -> PyResult<Self> {
        let index_path = index_path_from_dir(&base_dir);
        let inner = SpannIndex::<f16, usize>::load_from_path(index_path).map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner = SpannIndex::<f16, usize>::load_from_path(path).map_err(runtime_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn epsilon_closure(&self) -> f32 {
        self.inner.epsilon_closure()
    }

    fn metadata(&self, id: usize) -> Option<usize> {
        self.inner.metadata(id).copied()
    }

    fn add_vector(&mut self, vector: Vec<f32>, metadata: usize) -> PyResult<usize> {
        let array = vector_to_array1_f16(vector, self.inner.dimension(), "vector")?;
        self.inner
            .add_vector_with_metadata(array, metadata)
            .map_err(runtime_err)
    }

    fn search(&self, query: Vec<f32>, k: usize, rng_factor: f32) -> PyResult<Vec<(usize, f32)>> {
        let array = vector_to_array1_f16(query, self.inner.dimension(), "query")?;
        Ok(self.inner.search(&array, k, rng_factor))
    }

    fn search_with_metadata(
        &self,
        query: Vec<f32>,
        k: usize,
        rng_factor: f32,
    ) -> PyResult<Vec<(usize, f32, usize)>> {
        let array = vector_to_array1_f16(query, self.inner.dimension(), "query")?;
        Ok(self
            .inner
            .search_with_metadata(&array, k, rng_factor)
            .into_iter()
            .map(|(id, dist, meta)| (id, dist, *meta))
            .collect())
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save_to_path(path).map_err(runtime_err)
    }

    fn save_to_dir(&self, base_dir: PathBuf) -> PyResult<()> {
        let index_path = index_path_from_dir(&base_dir);
        self.inner.save_to_path(index_path).map_err(runtime_err)
    }
}

#[pymodule]
fn spann(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", CARGO_PKG_VERSION)?;
    module.add_function(wrap_pyfunction!(cargo_version, module)?)?;
    module.add_class::<SpannIndexF32>()?;
    module.add_class::<SpannIndexF32Meta>()?;
    module.add_class::<SpannIndexF16>()?;
    module.add_class::<SpannIndexF16Meta>()?;
    Ok(())
}
