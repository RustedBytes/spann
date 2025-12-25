use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected;
#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

use half::f16;
use ndarray::{Array1, Zip};

/// Type alias for high-dimensional vectors backed by ndarray.
pub type Vector<T> = Array1<T>;

pub trait Scalar: Copy {
    const BYTE_SIZE: usize;
    const FILE_TAG: &'static str;
    fn to_le_bytes(self) -> Vec<u8>;
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn from_f32(value: f32) -> Self;
    fn into_f32(self) -> f32;
    fn squared_euclidean(a: &Vector<Self>, b: &Vector<Self>) -> f32 {
        Zip::from(a).and(b).fold(0.0, |acc, &x, &y| {
            let diff = x.into_f32() - y.into_f32();
            acc + diff * diff
        })
    }
}

impl Scalar for f32 {
    const BYTE_SIZE: usize = std::mem::size_of::<f32>();
    const FILE_TAG: &'static str = "f32";

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let bytes: [u8; 4] = bytes.try_into().expect("Expected 4 bytes for f32");
        Self::from_le_bytes(bytes)
    }

    fn from_f32(value: f32) -> Self {
        value
    }

    fn into_f32(self) -> f32 {
        self
    }

    fn squared_euclidean(a: &Vector<Self>, b: &Vector<Self>) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let a_slice = a.as_slice().expect("Array1 must be contiguous");
            let b_slice = b.as_slice().expect("Array1 must be contiguous");

            if is_x86_feature_detected!("avx2") {
                unsafe { return squared_euclidean_f32_avx2(a_slice, b_slice) };
            }
            if is_x86_feature_detected!("sse2") {
                unsafe { return squared_euclidean_f32_sse2(a_slice, b_slice) };
            }
        }

        Zip::from(a).and(b).fold(0.0, |acc, &x, &y| {
            let diff = x - y;
            acc + diff * diff
        })
    }
}

impl Scalar for f16 {
    const BYTE_SIZE: usize = std::mem::size_of::<f16>();
    const FILE_TAG: &'static str = "f16";

    fn to_le_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let bytes: [u8; 2] = bytes.try_into().expect("Expected 2 bytes for f16");
        Self::from_le_bytes(bytes)
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }

    fn into_f32(self) -> f32 {
        self.to_f32()
    }
}

/// Newtype for enforcing type safety on IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VectorId(usize);

/// A simplified representation of the "Disk" storage.
/// This version writes vectors into batch files and reads on demand.
struct DataStore<T: Scalar> {
    dir: PathBuf,
    dimension: usize,
    count: usize,
    vectors_per_file: usize,
    cleanup_on_drop: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> DataStore<T> {
    const DEFAULT_VECTORS_PER_FILE: usize = 4096;

    fn from_vectors(dimension: usize, vectors: &[Vector<T>]) -> Result<Self, String> {
        Self::from_vectors_with_batch(dimension, vectors, Self::DEFAULT_VECTORS_PER_FILE)
    }

    fn from_vectors_with_batch(
        dimension: usize,
        vectors: &[Vector<T>],
        vectors_per_file: usize,
    ) -> Result<Self, String> {
        Self::validate_vectors(dimension, vectors)?;
        Self::validate_vectors_per_file(vectors_per_file)?;

        let mut dir = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pid = std::process::id();
        dir.push(format!("spann_store_{pid}_{nanos}_vectors_{}", T::FILE_TAG));

        std::fs::create_dir_all(&dir).map_err(|err| {
            format!(
                "Failed to create temporary data store directory {}: {err}",
                dir.display()
            )
        })?;

        Self::write_vectors(&dir, vectors, vectors_per_file)?;

        Ok(Self {
            dir,
            dimension,
            count: vectors.len(),
            vectors_per_file,
            cleanup_on_drop: true,
            _marker: std::marker::PhantomData,
        })
    }

    fn from_vectors_in_dir(
        dimension: usize,
        vectors: &[Vector<T>],
        base_dir: &Path,
    ) -> Result<Self, String> {
        Self::from_vectors_in_dir_with_batch(
            dimension,
            vectors,
            base_dir,
            Self::DEFAULT_VECTORS_PER_FILE,
        )
    }

    fn from_vectors_in_dir_with_batch(
        dimension: usize,
        vectors: &[Vector<T>],
        base_dir: &Path,
        vectors_per_file: usize,
    ) -> Result<Self, String> {
        Self::validate_vectors(dimension, vectors)?;
        Self::validate_vectors_per_file(vectors_per_file)?;

        std::fs::create_dir_all(base_dir).map_err(|err| {
            format!(
                "Failed to create data store directory {}: {err}",
                base_dir.display()
            )
        })?;

        let dir = base_dir.join(format!("vectors_{}", T::FILE_TAG));
        let record_len = dimension
            .checked_mul(T::BYTE_SIZE)
            .ok_or_else(|| "Vector dimension is too large".to_string())?;

        std::fs::create_dir_all(&dir).map_err(|err| {
            format!(
                "Failed to create data store directory {}: {err}",
                dir.display()
            )
        })?;

        let existing_batches = Self::list_batch_ids(&dir)?;
        if !existing_batches.is_empty() {
            let count = Self::count_vectors_in_batches(
                &dir,
                &existing_batches,
                record_len,
                vectors_per_file,
            )?;
            if count != vectors.len() {
                return Err(format!(
                    "Data store has {count} vectors but {expected} were provided",
                    expected = vectors.len()
                ));
            }

            return Ok(Self {
                dir,
                dimension,
                count,
                vectors_per_file,
                cleanup_on_drop: false,
                _marker: std::marker::PhantomData,
            });
        }

        Self::write_vectors(&dir, vectors, vectors_per_file)?;

        Ok(Self {
            dir,
            dimension,
            count: vectors.len(),
            vectors_per_file,
            cleanup_on_drop: false,
            _marker: std::marker::PhantomData,
        })
    }

    fn open_from_vectors_dir(
        dimension: usize,
        vectors_dir: impl AsRef<Path>,
        vectors_per_file: usize,
    ) -> Result<Self, String> {
        if dimension == 0 {
            return Err("Vector dimension must be > 0".to_string());
        }
        Self::validate_vectors_per_file(vectors_per_file)?;

        let dir = vectors_dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(format!(
                "Vector store directory {} does not exist",
                dir.display()
            ));
        }

        let record_len = dimension
            .checked_mul(T::BYTE_SIZE)
            .ok_or_else(|| "Vector dimension is too large".to_string())?;
        let batch_ids = Self::list_batch_ids(&dir)?;
        if batch_ids.is_empty() {
            return Err(format!(
                "Vector store directory {} has no batch files",
                dir.display()
            ));
        }
        let count = Self::count_vectors_in_batches(&dir, &batch_ids, record_len, vectors_per_file)?;

        Ok(Self {
            dir,
            dimension,
            count,
            vectors_per_file,
            cleanup_on_drop: false,
            _marker: std::marker::PhantomData,
        })
    }

    fn get(&self, id: VectorId) -> Vector<T> {
        assert!(
            id.0 < self.count,
            "VectorId {} out of bounds (count {})",
            id.0,
            self.count
        );

        let record_len = self.record_len();
        let batch_id = id.0 / self.vectors_per_file;
        let offset = (id.0 % self.vectors_per_file) * record_len;
        let path = self.batch_path(batch_id);
        let mut file = File::open(&path)
            .unwrap_or_else(|err| panic!("Failed to open batch file {}: {err}", path.display()));

        if let Err(err) = file.seek(SeekFrom::Start(offset as u64)) {
            panic!(
                "Failed to seek to vector {id:?} in {}: {err}",
                path.display()
            );
        }

        let mut buffer = vec![0u8; record_len];
        if let Err(err) = file.read_exact(&mut buffer) {
            panic!(
                "Failed to read vector {id:?} from {}: {err}",
                path.display()
            );
        }

        let mut values = Vec::with_capacity(self.dimension);
        for chunk in buffer.chunks_exact(T::BYTE_SIZE) {
            values.push(T::from_le_bytes(chunk));
        }
        Array1::from(values)
    }

    fn validate_vectors(dimension: usize, vectors: &[Vector<T>]) -> Result<(), String> {
        if dimension == 0 {
            return Err("Vector dimension must be > 0".to_string());
        }

        for (idx, vector) in vectors.iter().enumerate() {
            if vector.len() != dimension {
                return Err(format!(
                    "Vector {idx} has dimension {}, expected {dimension}",
                    vector.len()
                ));
            }
        }

        Ok(())
    }

    fn validate_vectors_per_file(vectors_per_file: usize) -> Result<(), String> {
        if vectors_per_file == 0 {
            return Err("vectors_per_file must be > 0".to_string());
        }
        Ok(())
    }

    fn append_vector(&mut self, vector: &Vector<T>) -> Result<VectorId, String> {
        self.validate_vector(vector)?;
        let id = self.count;
        let record_len = self.record_len();
        let batch_id = id / self.vectors_per_file;
        let offset = (id % self.vectors_per_file) * record_len;
        let path = self.batch_path(batch_id);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .map_err(|err| format!("Failed to open batch file {}: {err}", path.display()))?;
        let file_len = file
            .metadata()
            .map_err(|err| format!("Failed to read {} metadata: {err}", path.display()))?
            .len() as usize;
        if file_len != offset {
            return Err(format!(
                "Batch file {} size {file_len} does not match expected {offset}",
                path.display()
            ));
        }
        if let Err(err) = file.seek(SeekFrom::Start(offset as u64)) {
            return Err(format!(
                "Failed to seek to offset {offset} in {}: {err}",
                path.display()
            ));
        }
        Self::write_vector_to_file(&mut file, vector)?;
        file.flush()
            .map_err(|err| format!("Failed to flush batch file {}: {err}", path.display()))?;
        self.count += 1;
        Ok(VectorId(id))
    }

    fn write_vectors(
        dir: &Path,
        vectors: &[Vector<T>],
        vectors_per_file: usize,
    ) -> Result<(), String> {
        for (batch_id, chunk) in vectors.chunks(vectors_per_file).enumerate() {
            let path = dir.join(Self::batch_filename(batch_id));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|err| format!("Failed to create batch file {}: {err}", path.display()))?;
            for vector in chunk {
                Self::write_vector_to_file(&mut file, vector)?;
            }
            file.flush()
                .map_err(|err| format!("Failed to flush batch file {}: {err}", path.display()))?;
        }
        Ok(())
    }

    fn validate_vector(&self, vector: &Vector<T>) -> Result<(), String> {
        if self.dimension == 0 {
            return Err("Vector dimension must be > 0".to_string());
        }
        if vector.len() != self.dimension {
            return Err(format!(
                "Vector has dimension {}, expected {}",
                vector.len(),
                self.dimension
            ));
        }
        Ok(())
    }

    fn write_vector_to_file(file: &mut File, vector: &Vector<T>) -> Result<(), String> {
        for &value in vector {
            let bytes = value.to_le_bytes();
            file.write_all(&bytes)
                .map_err(|err| format!("Failed to write vector data: {err}"))?;
        }
        Ok(())
    }

    fn record_len(&self) -> usize {
        self.dimension
            .checked_mul(T::BYTE_SIZE)
            .expect("Vector dimension is too large")
    }

    fn batch_filename(batch_id: usize) -> String {
        format!("batch_{batch_id}.bin")
    }

    fn batch_path(&self, batch_id: usize) -> PathBuf {
        self.dir.join(Self::batch_filename(batch_id))
    }

    fn parse_batch_filename(name: &str) -> Option<usize> {
        let name = name.strip_prefix("batch_")?;
        let name = name.strip_suffix(".bin")?;
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        name.parse().ok()
    }

    fn list_batch_ids(dir: &Path) -> Result<Vec<usize>, String> {
        let mut ids = Vec::new();
        let entries = std::fs::read_dir(dir).map_err(|err| {
            format!(
                "Failed to read data store directory {}: {err}",
                dir.display()
            )
        })?;

        for entry in entries {
            let entry = entry.map_err(|err| {
                format!(
                    "Failed to read entry in data store {}: {err}",
                    dir.display()
                )
            })?;
            let file_type = entry
                .file_type()
                .map_err(|err| format!("Failed to read entry type: {err}"))?;
            if !file_type.is_file() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(id) = Self::parse_batch_filename(&name) {
                ids.push(id);
            }
        }

        ids.sort_unstable();
        Ok(ids)
    }

    fn count_vectors_in_batches(
        dir: &Path,
        batch_ids: &[usize],
        record_len: usize,
        vectors_per_file: usize,
    ) -> Result<usize, String> {
        for (expected, id) in batch_ids.iter().enumerate() {
            if *id != expected {
                return Err(format!(
                    "Data store is missing batch file for id {expected}"
                ));
            }
        }

        let mut total = 0usize;
        let last_index = batch_ids.len().saturating_sub(1);
        for (idx, batch_id) in batch_ids.iter().enumerate() {
            let path = dir.join(Self::batch_filename(*batch_id));
            let file_len = std::fs::metadata(&path)
                .map_err(|err| format!("Failed to read {} metadata: {err}", path.display()))?
                .len() as usize;
            if file_len == 0 {
                return Err(format!("Batch file {} is empty", path.display()));
            }
            if !file_len.is_multiple_of(record_len) {
                return Err(format!(
                    "Batch file {} size {file_len} is not a multiple of {record_len}",
                    path.display()
                ));
            }
            let count = file_len / record_len;
            if count > vectors_per_file {
                return Err(format!(
                    "Batch file {} holds {count} vectors, exceeds max {vectors_per_file}",
                    path.display()
                ));
            }
            if idx < last_index && count != vectors_per_file {
                return Err(format!(
                    "Batch file {} holds {count} vectors, expected {vectors_per_file}",
                    path.display()
                ));
            }
            total = total
                .checked_add(count)
                .ok_or_else(|| "Vector count overflowed".to_string())?;
        }

        Ok(total)
    }
}

impl<T: Scalar> Drop for DataStore<T> {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

/// Represents a posting list associated with a centroid.
struct PostingList<T: Scalar> {
    centroid: Vector<T>,
    // Stores IDs of vectors belonging to this cluster.
    // In SPANN, this list resides on disk.
    members: Vec<VectorId>,
}

pub struct SpannIndex<T: Scalar, M = ()> {
    dimension: usize,
    /// Centroids kept in memory for fast coarse-grained search.
    posting_lists: Vec<PostingList<T>>,
    /// The "disk" storage containing the actual full vectors.
    store: DataStore<T>,
    /// Metadata aligned with vector IDs (index 0 matches vector 0).
    metadata: Vec<M>,
    /// Hyperparameter: epsilon for soft assignment (closure).
    epsilon_closure: f32,
}

#[derive(Debug, Clone)]
struct SearchResult {
    id: VectorId,
    distance: f32,
}

// Custom ordering for MinHeap (BinaryHeap is MaxHeap by default)
impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for SearchResult {}
impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for MinHeap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Scalar, M> SpannIndex<T, M> {
    /// Returns the vector dimension this index was built for.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the epsilon closure hyperparameter used during build.
    pub fn epsilon_closure(&self) -> f32 {
        self.epsilon_closure
    }

    /// Returns metadata for a vector ID, if it exists.
    pub fn metadata(&self, id: usize) -> Option<&M> {
        self.metadata.get(id)
    }

    /// Adds a new vector to the store and assigns it to posting lists.
    /// Centroids are kept fixed, so this is a lightweight incremental path.
    pub fn add_vector_with_metadata(
        &mut self,
        vector: Vector<T>,
        metadata: M,
    ) -> Result<usize, String> {
        if vector.len() != self.dimension {
            return Err(format!(
                "Vector has dimension {}, expected {}",
                vector.len(),
                self.dimension
            ));
        }

        let vid = self.store.append_vector(&vector)?;
        let distances: Vec<(usize, f32)> = self
            .posting_lists
            .iter()
            .enumerate()
            .map(|(i, pl)| (i, squared_euclidean(&vector, &pl.centroid)))
            .collect();

        let min_dist = distances
            .iter()
            .map(|(_, d)| *d)
            .fold(f32::INFINITY, |a, b| a.min(b));
        let threshold = min_dist * (1.0 + self.epsilon_closure).powi(2);

        for (i, dist) in distances {
            if dist <= threshold {
                self.posting_lists[i].members.push(vid);
            }
        }

        self.metadata.push(metadata);
        Ok(vid.0)
    }

    /// Initialize index with raw data, metadata, and number of centroids (k).
    /// Uses a simplified K-Means for centroid selection.
    pub fn build_with_metadata(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
        metadata: Vec<M>,
        k_centroids: usize,
        epsilon_closure: f32,
    ) -> Result<Self, String> {
        Self::build_with_store(
            dimension,
            raw_data,
            metadata,
            k_centroids,
            epsilon_closure,
            DataStore::from_vectors,
        )
    }

    /// Initialize index with raw data, metadata, and number of centroids (k),
    /// while customizing how many vectors are stored per batch file.
    pub fn build_with_metadata_and_vectors_per_file(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
        metadata: Vec<M>,
        k_centroids: usize,
        epsilon_closure: f32,
        vectors_per_file: usize,
    ) -> Result<Self, String> {
        Self::build_with_store(
            dimension,
            raw_data,
            metadata,
            k_centroids,
            epsilon_closure,
            move |dim, data| DataStore::from_vectors_with_batch(dim, data, vectors_per_file),
        )
    }

    /// Initialize index with raw data, metadata, and number of centroids (k),
    /// while specifying where the on-disk store should live.
    pub fn build_with_metadata_and_store_dir(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
        metadata: Vec<M>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let store_dir = store_dir.as_ref().to_path_buf();
        Self::build_with_store(
            dimension,
            raw_data,
            metadata,
            k_centroids,
            epsilon_closure,
            move |dim, data| DataStore::from_vectors_in_dir(dim, data, &store_dir),
        )
    }

    /// Initialize index with raw data, metadata, and number of centroids (k),
    /// while specifying where the on-disk store should live and the batch size.
    pub fn build_with_metadata_and_store_dir_and_batch(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
        metadata: Vec<M>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: impl AsRef<Path>,
        vectors_per_file: usize,
    ) -> Result<Self, String> {
        let store_dir = store_dir.as_ref().to_path_buf();
        Self::build_with_store(
            dimension,
            raw_data,
            metadata,
            k_centroids,
            epsilon_closure,
            move |dim, data| {
                DataStore::from_vectors_in_dir_with_batch(dim, data, &store_dir, vectors_per_file)
            },
        )
    }

    fn build_with_store<F>(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
        metadata: Vec<M>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_builder: F,
    ) -> Result<Self, String>
    where
        F: FnOnce(usize, &[Vector<T>]) -> Result<DataStore<T>, String>,
    {
        if raw_data.is_empty() {
            return Err("Cannot build index with empty data".to_string());
        }
        if metadata.len() != raw_data.len() {
            return Err(format!(
                "Metadata has {} entries but {} vectors were provided",
                metadata.len(),
                raw_data.len()
            ));
        }

        let mut centroids: Vec<Vector<T>> = raw_data.iter().take(k_centroids).cloned().collect();

        for _ in 0..10 {
            let mut sums: Vec<Array1<f32>> =
                (0..k_centroids).map(|_| Array1::zeros(dimension)).collect();
            let mut counts = vec![0; k_centroids];

            for point in &raw_data {
                let closest_idx = Self::find_closest_centroid(point, &centroids).0;
                Zip::from(&mut sums[closest_idx])
                    .and(point)
                    .for_each(|sum, &val| {
                        *sum += val.into_f32();
                    });
                counts[closest_idx] += 1;
            }

            for (j, centroid) in centroids.iter_mut().enumerate() {
                if counts[j] > 0 {
                    let denom = counts[j] as f32;
                    Zip::from(centroid).and(&sums[j]).for_each(|c, &s| {
                        *c = T::from_f32(s / denom);
                    });
                }
            }
        }

        let mut posting_lists: Vec<PostingList<T>> = centroids
            .into_iter()
            .map(|c| PostingList {
                centroid: c,
                members: Vec::new(),
            })
            .collect();

        for (idx, point) in raw_data.iter().enumerate() {
            let vid = VectorId(idx);

            let distances: Vec<(usize, f32)> = posting_lists
                .iter()
                .enumerate()
                .map(|(i, pl)| (i, squared_euclidean(point, &pl.centroid)))
                .collect();

            let min_dist = distances
                .iter()
                .map(|(_, d)| *d)
                .fold(f32::INFINITY, |a, b| a.min(b));

            let threshold = min_dist * (1.0 + epsilon_closure).powi(2);

            for (i, dist) in distances {
                if dist <= threshold {
                    posting_lists[i].members.push(vid);
                }
            }
        }

        let store = store_builder(dimension, &raw_data)?;

        Ok(Self {
            dimension,
            posting_lists,
            store,
            metadata,
            epsilon_closure,
        })
    }

    /// Helper to find the index of the nearest centroid
    fn find_closest_centroid(point: &Vector<T>, centroids: &[Vector<T>]) -> (usize, f32) {
        let mut min_dist = f32::INFINITY;
        let mut best_idx = 0;

        for (i, c) in centroids.iter().enumerate() {
            let d = squared_euclidean(point, c);
            if d < min_dist {
                min_dist = d;
                best_idx = i;
            }
        }
        (best_idx, min_dist)
    }

    /// Search the index for the k nearest neighbors.
    /// Uses 'rng_factor' (alpha in paper) to dynamically prune centroids.
    pub fn search(&self, query: &Vector<T>, k: usize, rng_factor: f32) -> Vec<(usize, f32)> {
        // 1. Identify relevant centroids (Coarse Search)
        // Calculate query distance to all centroids
        let mut centroid_dists: Vec<(usize, f32)> = self
            .posting_lists
            .iter()
            .enumerate()
            .map(|(i, pl)| (i, squared_euclidean(query, &pl.centroid)))
            .collect();

        // Sort by distance ascending
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if centroid_dists.is_empty() {
            return vec![];
        }

        let nearest_centroid_dist = centroid_dists[0].1;
        let threshold = nearest_centroid_dist * (1.0 + rng_factor).powi(2);

        // 2. Select Posting Lists to probe
        // Pruning: Only check centroids within (1 + alpha) of the nearest.
        let candidates: Vec<usize> = centroid_dists
            .into_iter()
            .take_while(|(_, d)| *d <= threshold)
            .map(|(idx, _)| idx)
            .collect();

        // 3. Fine-grained Search (Scan selected lists)
        let mut min_heap = BinaryHeap::new();
        // Use a simple specialized HashSet for visited if duplication is high,
        // but typically raw iteration with post-dedup or a bitset is faster for small batches.
        // Here we assume vectors might be duplicated across lists due to soft assignment,
        // so we need deduplication.
        let mut visited_ids = std::collections::HashSet::new();

        for list_idx in candidates {
            let list = &self.posting_lists[list_idx];

            for &vid in &list.members {
                if visited_ids.contains(&vid) {
                    continue;
                }
                visited_ids.insert(vid);

                let vec_data = self.store.get(vid);
                let dist = squared_euclidean(query, &vec_data);

                if min_heap.len() < k {
                    min_heap.push(SearchResult {
                        id: vid,
                        distance: dist,
                    });
                } else if let Some(top) = min_heap.peek()
                    && dist < top.distance
                {
                    min_heap.pop();
                    min_heap.push(SearchResult {
                        id: vid,
                        distance: dist,
                    });
                }
            }
        }

        // Convert back to sorted vector
        let mut results: Vec<(usize, f32)> = min_heap
            .into_sorted_vec() // Returns ascending order (min first) usually?
            // Actually into_sorted_vec returns ascending order for a MaxHeap.
            // Our SearchResult is ordered such that "Greater" means "Smaller Distance" (for MinHeap behavior).
            // So into_sorted_vec gives us largest distances first (worst matches).
            .into_iter()
            .map(|r| (r.id.0, r.distance))
            .collect();

        // Reverse to get Best (Smallest Dist) -> Worst
        results.reverse();
        results
    }

    /// Search the index and return metadata aligned with each result.
    pub fn search_with_metadata<'a>(
        &'a self,
        query: &Vector<T>,
        k: usize,
        rng_factor: f32,
    ) -> Vec<(usize, f32, &'a M)> {
        self.search(query, k, rng_factor)
            .into_iter()
            .map(|(id, dist)| {
                let meta = self
                    .metadata
                    .get(id)
                    .expect("Missing metadata for vector id");
                (id, dist, meta)
            })
            .collect()
    }
}

impl<T: Scalar> SpannIndex<T, ()> {
    /// Initialize index with raw data and number of centroids (k).
    /// Uses a simplified K-Means for centroid selection.
    pub fn build(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
        k_centroids: usize,
        epsilon_closure: f32,
    ) -> Result<Self, String> {
        let metadata = vec![(); raw_data.len()];
        Self::build_with_metadata(dimension, raw_data, metadata, k_centroids, epsilon_closure)
    }

    /// Initialize index with raw data and number of centroids (k),
    /// while customizing how many vectors are stored per batch file.
    pub fn build_with_vectors_per_file(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
        k_centroids: usize,
        epsilon_closure: f32,
        vectors_per_file: usize,
    ) -> Result<Self, String> {
        let metadata = vec![(); raw_data.len()];
        Self::build_with_metadata_and_vectors_per_file(
            dimension,
            raw_data,
            metadata,
            k_centroids,
            epsilon_closure,
            vectors_per_file,
        )
    }

    /// Initialize index with raw data and number of centroids (k),
    /// while specifying where the on-disk store should live.
    pub fn build_with_store_dir(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let metadata = vec![(); raw_data.len()];
        Self::build_with_metadata_and_store_dir(
            dimension,
            raw_data,
            metadata,
            k_centroids,
            epsilon_closure,
            store_dir,
        )
    }

    /// Initialize index with raw data and number of centroids (k),
    /// while specifying where the on-disk store should live and the batch size.
    pub fn build_with_store_dir_and_batch(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
        k_centroids: usize,
        epsilon_closure: f32,
        store_dir: impl AsRef<Path>,
        vectors_per_file: usize,
    ) -> Result<Self, String> {
        let metadata = vec![(); raw_data.len()];
        Self::build_with_metadata_and_store_dir_and_batch(
            dimension,
            raw_data,
            metadata,
            k_centroids,
            epsilon_closure,
            store_dir,
            vectors_per_file,
        )
    }

    /// Adds a new vector to the store and assigns it to posting lists.
    /// Centroids are kept fixed, so this is a lightweight incremental path.
    pub fn add_vector(&mut self, vector: Vector<T>) -> Result<usize, String> {
        self.add_vector_with_metadata(vector, ())
    }
}

impl<T: Scalar> SpannIndex<T, usize> {
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let mut file = File::create(path.as_ref()).map_err(|err| {
            format!(
                "Failed to create index file {}: {err}",
                path.as_ref().display()
            )
        })?;

        write_exact(&mut file, INDEX_MAGIC)?;
        write_u32(&mut file, INDEX_VERSION)?;
        write_string(&mut file, T::FILE_TAG)?;
        write_u64(&mut file, self.dimension as u64)?;
        write_f32(&mut file, self.epsilon_closure)?;
        write_u64(&mut file, self.store.vectors_per_file as u64)?;
        write_u64(&mut file, self.store.count as u64)?;
        let store_dir = std::fs::canonicalize(&self.store.dir).unwrap_or(self.store.dir.clone());
        write_string(&mut file, &store_dir.to_string_lossy())?;
        write_u64(&mut file, self.posting_lists.len() as u64)?;

        for list in &self.posting_lists {
            for &value in &list.centroid {
                let bytes = value.to_le_bytes();
                write_exact(&mut file, &bytes)?;
            }

            write_u64(&mut file, list.members.len() as u64)?;
            for id in &list.members {
                write_u64(&mut file, id.0 as u64)?;
            }
        }

        write_u64(&mut file, self.metadata.len() as u64)?;
        for &meta in &self.metadata {
            write_u64(&mut file, meta as u64)?;
        }

        Ok(())
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let mut file = File::open(path)
            .map_err(|err| format!("Failed to open index file {}: {err}", path.display()))?;

        let mut magic = [0u8; INDEX_MAGIC.len()];
        read_exact(&mut file, &mut magic)?;
        if &magic != INDEX_MAGIC {
            return Err(format!(
                "Index file {} has invalid magic bytes",
                path.display()
            ));
        }

        let version = read_u32(&mut file)?;
        if version != INDEX_VERSION {
            return Err(format!(
                "Index file {} has unsupported version {}",
                path.display(),
                version
            ));
        }

        let tag = read_string(&mut file)?;
        if tag != T::FILE_TAG {
            return Err(format!(
                "Index file {} expects scalar type {}, got {}",
                path.display(),
                T::FILE_TAG,
                tag
            ));
        }

        let dimension = to_usize(read_u64(&mut file)?, "dimension")?;
        let epsilon_closure = read_f32(&mut file)?;
        let vectors_per_file = to_usize(read_u64(&mut file)?, "vectors_per_file")?;
        let expected_count = to_usize(read_u64(&mut file)?, "expected_count")?;
        let store_dir_raw = read_string(&mut file)?;
        let store_dir = resolve_index_path(path, &store_dir_raw);

        let posting_list_count = to_usize(read_u64(&mut file)?, "posting_list_count")?;
        let mut posting_lists = Vec::with_capacity(posting_list_count);
        for _ in 0..posting_list_count {
            let mut centroid_values = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                let mut buffer = vec![0u8; T::BYTE_SIZE];
                read_exact(&mut file, &mut buffer)?;
                centroid_values.push(T::from_le_bytes(&buffer));
            }
            let centroid = Array1::from(centroid_values);

            let member_count = to_usize(read_u64(&mut file)?, "member_count")?;
            let mut members = Vec::with_capacity(member_count);
            for _ in 0..member_count {
                let id = to_usize(read_u64(&mut file)?, "member_id")?;
                members.push(VectorId(id));
            }
            posting_lists.push(PostingList { centroid, members });
        }

        let metadata_len = to_usize(read_u64(&mut file)?, "metadata_len")?;
        let mut metadata = Vec::with_capacity(metadata_len);
        for _ in 0..metadata_len {
            let meta = to_usize(read_u64(&mut file)?, "metadata_value")?;
            metadata.push(meta);
        }

        if metadata.len() != expected_count {
            return Err(format!(
                "Index file {} metadata count {} does not match expected {expected_count}",
                path.display(),
                metadata.len()
            ));
        }

        for list in &posting_lists {
            for id in &list.members {
                if id.0 >= expected_count {
                    return Err(format!(
                        "Index file {} has member id {} out of bounds",
                        path.display(),
                        id.0
                    ));
                }
            }
        }

        let store = DataStore::open_from_vectors_dir(dimension, store_dir, vectors_per_file)?;
        if store.count != expected_count {
            return Err(format!(
                "Index file {} expects {expected_count} vectors, store has {}",
                path.display(),
                store.count
            ));
        }

        Ok(Self {
            dimension,
            posting_lists,
            store,
            metadata,
            epsilon_closure,
        })
    }
}

const INDEX_MAGIC: &[u8; 8] = b"SPANNIDX";
const INDEX_VERSION: u32 = 1;

fn write_exact(writer: &mut File, bytes: &[u8]) -> Result<(), String> {
    writer
        .write_all(bytes)
        .map_err(|err| format!("Failed to write index data: {err}"))
}

fn read_exact(reader: &mut File, buffer: &mut [u8]) -> Result<(), String> {
    reader
        .read_exact(buffer)
        .map_err(|err| format!("Failed to read index data: {err}"))
}

fn write_u32(writer: &mut File, value: u32) -> Result<(), String> {
    write_exact(writer, &value.to_le_bytes())
}

fn read_u32(reader: &mut File) -> Result<u32, String> {
    let mut buffer = [0u8; 4];
    read_exact(reader, &mut buffer)?;
    Ok(u32::from_le_bytes(buffer))
}

fn write_u64(writer: &mut File, value: u64) -> Result<(), String> {
    write_exact(writer, &value.to_le_bytes())
}

fn read_u64(reader: &mut File) -> Result<u64, String> {
    let mut buffer = [0u8; 8];
    read_exact(reader, &mut buffer)?;
    Ok(u64::from_le_bytes(buffer))
}

fn write_f32(writer: &mut File, value: f32) -> Result<(), String> {
    write_exact(writer, &value.to_le_bytes())
}

fn read_f32(reader: &mut File) -> Result<f32, String> {
    let mut buffer = [0u8; 4];
    read_exact(reader, &mut buffer)?;
    Ok(f32::from_le_bytes(buffer))
}

fn write_string(writer: &mut File, value: &str) -> Result<(), String> {
    write_u64(writer, value.len() as u64)?;
    write_exact(writer, value.as_bytes())
}

fn read_string(reader: &mut File) -> Result<String, String> {
    let len = read_u64(reader)? as usize;
    let mut buffer = vec![0u8; len];
    read_exact(reader, &mut buffer)?;
    String::from_utf8(buffer).map_err(|err| format!("Invalid UTF-8 in index file: {err}"))
}

fn resolve_index_path(index_path: &Path, stored: &str) -> PathBuf {
    let stored_path = PathBuf::from(stored);
    if stored_path.is_relative() {
        let base = index_path.parent().unwrap_or_else(|| Path::new("."));
        base.join(stored_path)
    } else {
        stored_path
    }
}

fn to_usize(value: u64, label: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("Index field {label} is too large"))
}

/// Computes Squared Euclidean Distance to avoid expensive SQRT operations during comparison.
#[inline(always)]
fn squared_euclidean<T: Scalar>(a: &Vector<T>, b: &Vector<T>) -> f32 {
    T::squared_euclidean(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn squared_euclidean_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = arch::_mm256_setzero_ps();
    let mut i = 0;
    let len = a.len();

    while i + 8 <= len {
        let va = arch::_mm256_loadu_ps(a.as_ptr().add(i));
        let vb = arch::_mm256_loadu_ps(b.as_ptr().add(i));
        let diff = arch::_mm256_sub_ps(va, vb);
        acc = arch::_mm256_add_ps(acc, arch::_mm256_mul_ps(diff, diff));
        i += 8;
    }

    let mut lanes = [0.0f32; 8];
    arch::_mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut total = lanes.iter().sum();

    while i < len {
        let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
        total += diff * diff;
        i += 1;
    }

    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn squared_euclidean_f32_sse2(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = arch::_mm_setzero_ps();
    let mut i = 0;
    let len = a.len();

    while i + 4 <= len {
        let va = arch::_mm_loadu_ps(a.as_ptr().add(i));
        let vb = arch::_mm_loadu_ps(b.as_ptr().add(i));
        let diff = arch::_mm_sub_ps(va, vb);
        acc = arch::_mm_add_ps(acc, arch::_mm_mul_ps(diff, diff));
        i += 4;
    }

    let mut lanes = [0.0f32; 4];
    arch::_mm_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut total = lanes.iter().sum();

    while i < len {
        let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
        total += diff * diff;
        i += 1;
    }

    total
}
