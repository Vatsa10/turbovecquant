//! TurboQuant implementation for vector search.
//!
//! Compresses high-dimensional vectors to 2-4 bits per coordinate with
//! near-optimal distortion. Data-oblivious — no training required.
//!
//! ```no_run
//! use turbovec::TurboQuantIndex;
//!
//! // 1536-dim vectors compressed to 4 bits per coordinate.
//! let mut index = TurboQuantIndex::new(1536, 4);
//!
//! // `vectors` is a flat [f32] of length n * dim, `queries` likewise.
//! let vectors: Vec<f32> = vec![0.0; 1536 * 10];
//! let queries: Vec<f32> = vec![0.0; 1536 * 2];
//!
//! index.add(&vectors);
//! let results = index.search(&queries, 10);
//! index.write("index.tv").unwrap();
//! let loaded = TurboQuantIndex::load("index.tv").unwrap();
//! ```
//!
//! # Concurrent search
//!
//! `search` takes `&self` and is safe to call from multiple threads
//! concurrently. Internally the rotation matrix, the Lloyd-Max centroids
//! and the SIMD-blocked code layout are initialised lazily via
//! [`std::sync::OnceLock`], so the first caller pays the one-time
//! initialisation cost and every subsequent caller reads the caches
//! without locking. [`TurboQuantIndex::prepare`] can be called once
//! after `add`/`load` to pay that cost up front.
//!
//! Mutation still flows through `&mut self`: `add` extends the packed
//! codes and invalidates the blocked layout cache by replacing its
//! `OnceLock`. This keeps the invariant that once a cache is populated
//! from `&self`, it matches the current `packed_codes`.

pub mod codebook;
pub mod encode;
pub mod io;
pub mod pack;
pub mod rotation;
pub mod search;

use std::path::Path;
use std::sync::OnceLock;

const ROTATION_SEED: u64 = 42;
const BLOCK: usize = 32;
const FLUSH_EVERY: usize = 256;

/// SIMD-blocked cache derived from `packed_codes`.
///
/// Materialised lazily by [`TurboQuantIndex::search`] on first call
/// and re-materialised when [`TurboQuantIndex::add`] resets the
/// enclosing `OnceLock`.
struct BlockedCache {
    data: Vec<u8>,
    n_blocks: usize,
}

pub struct TurboQuantIndex {
    dim: usize,
    bit_width: usize,
    n_vectors: usize,
    packed_codes: Vec<u8>,
    norms: Vec<f32>,

    // Tombstones: one byte per slot, 1 = deleted. Parallel to `norms`,
    // always resized in lockstep with `n_vectors`. `free_list` lets
    // `add` reuse a previously-deleted slot instead of appending.
    tombstones: Vec<u8>,
    free_list: Vec<usize>,
    num_deleted: usize,

    // Thread-safe lazy caches. These are initialised from `&self` via
    // `OnceLock::get_or_init`, which allows `search` to take `&self`
    // and run concurrently from multiple threads without external
    // locking. `add` resets `blocked` by replacing its `OnceLock` (it
    // already has `&mut self` for the underlying extend on
    // `packed_codes` and `norms`).
    //
    // `rotation` and `centroids` are deterministic functions of `(dim,
    // ROTATION_SEED)` and `(bit_width, dim)` respectively, so they
    // never need to be invalidated.
    rotation: OnceLock<Vec<f32>>,
    centroids: OnceLock<Vec<f32>>,
    blocked: OnceLock<BlockedCache>,
}

pub struct SearchResults {
    pub scores: Vec<f32>,
    pub indices: Vec<i64>,
    pub nq: usize,
    pub k: usize,
}

impl SearchResults {
    pub fn scores_for_query(&self, qi: usize) -> &[f32] {
        &self.scores[qi * self.k..(qi + 1) * self.k]
    }

    pub fn indices_for_query(&self, qi: usize) -> &[i64] {
        &self.indices[qi * self.k..(qi + 1) * self.k]
    }
}

impl TurboQuantIndex {
    pub fn new(dim: usize, bit_width: usize) -> Self {
        assert!((2..=4).contains(&bit_width), "bit_width must be 2, 3, or 4");
        assert!(dim % 8 == 0, "dim must be a multiple of 8");

        Self {
            dim,
            bit_width,
            n_vectors: 0,
            packed_codes: Vec::new(),
            norms: Vec::new(),
            tombstones: Vec::new(),
            free_list: Vec::new(),
            num_deleted: 0,
            rotation: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
        }
    }

    #[inline]
    fn bytes_per_vector(&self) -> usize {
        (self.dim / 8) * self.bit_width
    }

    /// Add vectors and return the slot ids they were written to.
    ///
    /// Slots previously freed by [`TurboQuantIndex::delete`] are reused
    /// in LIFO order; any remainder is appended. The returned vector
    /// has length equal to the number of input vectors and is in
    /// input order, so callers can pair each input with its slot id.
    pub fn add(&mut self, vectors: &[f32]) -> Vec<i64> {
        let n = vectors.len() / self.dim;
        assert_eq!(
            vectors.len(),
            n * self.dim,
            "vectors length must be a multiple of dim"
        );
        if n == 0 {
            return Vec::new();
        }

        let rotation = self
            .rotation
            .get_or_init(|| rotation::make_rotation_matrix(self.dim));
        let (boundaries, _) = codebook::codebook(self.bit_width, self.dim);
        let (packed, norms) =
            encode::encode(vectors, n, self.dim, rotation, &boundaries, self.bit_width);

        let bpv = self.bytes_per_vector();
        let mut slots = Vec::with_capacity(n);
        let mut i = 0;

        // Reuse deleted slots first. Writing into `packed_codes` directly
        // is safe because the per-vector layout is contiguous: slot `s`
        // owns bytes `s * bpv .. (s + 1) * bpv`.
        while i < n {
            let Some(slot) = self.free_list.pop() else {
                break;
            };
            let src = i * bpv;
            let dst = slot * bpv;
            self.packed_codes[dst..dst + bpv].copy_from_slice(&packed[src..src + bpv]);
            self.norms[slot] = norms[i];
            self.tombstones[slot] = 0;
            self.num_deleted -= 1;
            slots.push(slot as i64);
            i += 1;
        }

        // Append whatever did not fit into a reused slot.
        if i < n {
            let rem = n - i;
            self.packed_codes.extend_from_slice(&packed[i * bpv..]);
            self.norms.extend_from_slice(&norms[i..]);
            self.tombstones.extend(std::iter::repeat(0u8).take(rem));
            for j in 0..rem {
                slots.push((self.n_vectors + j) as i64);
            }
            self.n_vectors += rem;
        }

        // Invalidate the blocked cache — it was derived from the old
        // `packed_codes` and no longer matches the new contents.
        self.blocked = OnceLock::new();
        slots
    }

    /// Mark one or more vectors as deleted.
    ///
    /// Deleted vectors are excluded from subsequent `search` results and
    /// their slots become available for reuse on the next `add`. This is
    /// a tombstone-based delete — storage is not compacted in place, so
    /// the on-disk and in-memory footprint is unchanged until slots are
    /// reused.
    ///
    /// Ids outside `[0, len())` and already-deleted ids are silently
    /// ignored, mirroring the convention that deletion is idempotent.
    pub fn delete(&mut self, ids: &[i64]) {
        for &id in ids {
            if id < 0 {
                continue;
            }
            let uid = id as usize;
            if uid >= self.n_vectors {
                continue;
            }
            if self.tombstones[uid] != 0 {
                continue;
            }
            self.tombstones[uid] = 1;
            self.free_list.push(uid);
            self.num_deleted += 1;
        }
        // Invalidate blocked cache so next search rebuilds (we don't
        // strictly need to — the SIMD scan still scores deleted slots —
        // but it keeps invariants simple and is cheap on next use).
        self.blocked = OnceLock::new();
    }

    pub fn is_deleted(&self, id: i64) -> bool {
        if id < 0 {
            return false;
        }
        let uid = id as usize;
        uid < self.n_vectors && self.tombstones[uid] != 0
    }

    pub fn num_deleted(&self) -> usize {
        self.num_deleted
    }

    /// Live vector count — excludes tombstoned slots.
    pub fn live_count(&self) -> usize {
        self.n_vectors - self.num_deleted
    }

    /// Physical slot capacity — including tombstoned slots.
    pub fn capacity(&self) -> usize {
        self.n_vectors
    }

    /// Run a top-`k` search against the index.
    ///
    /// Takes `&self` and is safe to call concurrently from multiple
    /// threads. The first caller on a fresh index pays the one-time
    /// cache initialisation cost (rotation matrix, Lloyd-Max centroids
    /// and the SIMD-blocked code layout). Subsequent callers read the
    /// caches without locking.
    ///
    /// Call [`TurboQuantIndex::prepare`] once after `add`/`load` to
    /// pay that cost up front if you want deterministic first-query
    /// latency.
    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        let nq = queries.len() / self.dim;
        assert_eq!(queries.len(), nq * self.dim);

        let rotation = self
            .rotation
            .get_or_init(|| rotation::make_rotation_matrix(self.dim));
        let centroids = self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bit_width, self.dim);
            c
        });
        let blocked = self.blocked.get_or_init(|| {
            let (data, n_blocks) =
                pack::repack(&self.packed_codes, self.n_vectors, self.bit_width, self.dim);
            BlockedCache { data, n_blocks }
        });

        let live = self.n_vectors - self.num_deleted;
        let k = k.min(live);

        // Over-fetch by `num_deleted` so that after dropping tombstoned
        // ids we still have at least `k` live results. When no vectors
        // are deleted this degenerates to the existing fast path.
        let k_search = (k + self.num_deleted).min(self.n_vectors);

        if k_search == 0 {
            return SearchResults {
                scores: vec![0.0; nq * k],
                indices: vec![-1; nq * k],
                nq,
                k,
            };
        }

        let (raw_scores, raw_indices) = search::search(
            queries,
            nq,
            rotation,
            &blocked.data,
            centroids,
            &self.norms,
            self.bit_width,
            self.dim,
            self.n_vectors,
            blocked.n_blocks,
            k_search,
        );

        if self.num_deleted == 0 {
            return SearchResults {
                scores: raw_scores,
                indices: raw_indices,
                nq,
                k,
            };
        }

        // Filter tombstones out of each query's top-k_search list,
        // keeping the first `k` live ids. Results arrive sorted best-to-
        // -worst so we just scan linearly.
        let mut scores = Vec::with_capacity(nq * k);
        let mut indices = Vec::with_capacity(nq * k);
        for qi in 0..nq {
            let row_start = qi * k_search;
            let row_end = row_start + k_search;
            let row_s = &raw_scores[row_start..row_end];
            let row_i = &raw_indices[row_start..row_end];
            let mut kept = 0;
            for (&s, &idx) in row_s.iter().zip(row_i.iter()) {
                if kept == k {
                    break;
                }
                if idx < 0 {
                    continue;
                }
                let uidx = idx as usize;
                if uidx < self.tombstones.len() && self.tombstones[uidx] == 0 {
                    scores.push(s);
                    indices.push(idx);
                    kept += 1;
                }
            }
            while kept < k {
                scores.push(0.0);
                indices.push(-1);
                kept += 1;
            }
        }

        SearchResults {
            scores,
            indices,
            nq,
            k,
        }
    }

    /// Eagerly populate the search caches (rotation matrix, centroids
    /// and SIMD-blocked code layout).
    ///
    /// Calling `prepare` is optional — `search` will materialise the
    /// caches on its first call if needed. Use it to move the one-time
    /// cost out of the first query path, for example right after
    /// [`TurboQuantIndex::load`] or after a batch of [`add`] calls.
    ///
    /// Safe to call multiple times and from multiple threads.
    pub fn prepare(&self) {
        self.rotation
            .get_or_init(|| rotation::make_rotation_matrix(self.dim));
        self.centroids.get_or_init(|| {
            let (_, c) = codebook::codebook(self.bit_width, self.dim);
            c
        });
        self.blocked.get_or_init(|| {
            let (data, n_blocks) =
                pack::repack(&self.packed_codes, self.n_vectors, self.bit_width, self.dim);
            BlockedCache { data, n_blocks }
        });
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        io::write(
            path,
            self.bit_width,
            self.dim,
            self.n_vectors,
            &self.packed_codes,
            &self.norms,
            &self.tombstones,
        )
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let (bit_width, dim, n_vectors, packed_codes, norms, tombstones) = io::load(path)?;
        let mut free_list = Vec::new();
        let mut num_deleted = 0usize;
        for (i, &t) in tombstones.iter().enumerate() {
            if t != 0 {
                free_list.push(i);
                num_deleted += 1;
            }
        }
        Ok(Self {
            dim,
            bit_width,
            n_vectors,
            packed_codes,
            norms,
            tombstones,
            free_list,
            num_deleted,
            rotation: OnceLock::new(),
            centroids: OnceLock::new(),
            blocked: OnceLock::new(),
        })
    }

    /// Live vector count — excludes tombstoned slots. This is the
    /// user-visible length (matches Python `len(index)`).
    pub fn len(&self) -> usize {
        self.n_vectors - self.num_deleted
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bit_width(&self) -> usize {
        self.bit_width
    }
}
