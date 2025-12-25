use std::cell::RefCell;
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
/// This version writes vectors to a file and reads them by offset.
struct DataStore<T: Scalar> {
    file: RefCell<File>,
    dimension: usize,
    count: usize,
    path: PathBuf,
    cleanup_on_drop: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> DataStore<T> {
    fn from_vectors(dimension: usize, vectors: &[Vector<T>]) -> Result<Self, String> {
        Self::validate_vectors(dimension, vectors)?;

        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pid = std::process::id();
        path.push(format!("spann_store_{pid}_{nanos}.bin"));

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|err| format!("Failed to create data store file: {err}"))?;

        Self::write_vectors(&mut file, vectors)?;

        Ok(Self {
            file: RefCell::new(file),
            dimension,
            count: vectors.len(),
            path,
            cleanup_on_drop: true,
            _marker: std::marker::PhantomData,
        })
    }

    fn from_vectors_in_dir(
        dimension: usize,
        vectors: &[Vector<T>],
        base_dir: &Path,
    ) -> Result<Self, String> {
        Self::validate_vectors(dimension, vectors)?;

        std::fs::create_dir_all(base_dir).map_err(|err| {
            format!(
                "Failed to create data store directory {}: {err}",
                base_dir.display()
            )
        })?;

        let path = base_dir.join(format!("vectors_{}.bin", T::FILE_TAG));
        let record_len = dimension
            .checked_mul(T::BYTE_SIZE)
            .ok_or_else(|| "Vector dimension is too large".to_string())?;

        if path.exists() {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .map_err(|err| format!("Failed to open data store file: {err}"))?;
            let file_len = file
                .metadata()
                .map_err(|err| format!("Failed to read data store metadata: {err}"))?
                .len() as usize;

            if record_len == 0 || !file_len.is_multiple_of(record_len) {
                return Err("Data store file size does not match dimension".to_string());
            }

            let count = file_len / record_len;
            if count != vectors.len() {
                return Err(format!(
                    "Data store has {count} vectors but {expected} were provided",
                    expected = vectors.len()
                ));
            }

            return Ok(Self {
                file: RefCell::new(file),
                dimension,
                count,
                path,
                cleanup_on_drop: false,
                _marker: std::marker::PhantomData,
            });
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|err| format!("Failed to create data store file: {err}"))?;

        Self::write_vectors(&mut file, vectors)?;

        Ok(Self {
            file: RefCell::new(file),
            dimension,
            count: vectors.len(),
            path,
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

        let record_len = self.dimension * T::BYTE_SIZE;
        let offset = id.0 as u64 * record_len as u64;
        let mut file = self.file.borrow_mut();

        if let Err(err) = file.seek(SeekFrom::Start(offset)) {
            panic!("Failed to seek to vector {id:?}: {err}");
        }

        let mut buffer = vec![0u8; record_len];
        if let Err(err) = file.read_exact(&mut buffer) {
            panic!("Failed to read vector {id:?}: {err}");
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

    fn write_vectors(file: &mut File, vectors: &[Vector<T>]) -> Result<(), String> {
        for vector in vectors {
            for &value in vector {
                let bytes = value.to_le_bytes();
                file.write_all(&bytes)
                    .map_err(|err| format!("Failed to write vector data: {err}"))?;
            }
        }

        file.flush()
            .map_err(|err| format!("Failed to flush vector data: {err}"))?;
        Ok(())
    }
}

impl<T: Scalar> Drop for DataStore<T> {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            let _ = std::fs::remove_file(&self.path);
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

pub struct SpannIndex<T: Scalar> {
    dimension: usize,
    /// Centroids kept in memory for fast coarse-grained search.
    posting_lists: Vec<PostingList<T>>,
    /// The "disk" storage containing the actual full vectors.
    store: DataStore<T>,
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

impl<T: Scalar> SpannIndex<T> {
    /// Returns the vector dimension this index was built for.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the epsilon closure hyperparameter used during build.
    pub fn epsilon_closure(&self) -> f32 {
        self.epsilon_closure
    }

    /// Initialize index with raw data and number of centroids (k).
    /// Uses a simplified K-Means for centroid selection.
    pub fn build(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
        k_centroids: usize,
        epsilon_closure: f32,
    ) -> Result<Self, String> {
        Self::build_with_store(
            dimension,
            raw_data,
            k_centroids,
            epsilon_closure,
            DataStore::from_vectors,
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
        let store_dir = store_dir.as_ref().to_path_buf();
        Self::build_with_store(
            dimension,
            raw_data,
            k_centroids,
            epsilon_closure,
            move |dim, data| DataStore::from_vectors_in_dir(dim, data, &store_dir),
        )
    }

    fn build_with_store<F>(
        dimension: usize,
        raw_data: Vec<Vector<T>>,
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
