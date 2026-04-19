use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyclass]
struct TurboQuantIndex {
    inner: turbovec_core::TurboQuantIndex,
}

#[pymethods]
impl TurboQuantIndex {
    #[new]
    fn new(dim: usize, bit_width: usize) -> Self {
        Self {
            inner: turbovec_core::TurboQuantIndex::new(dim, bit_width),
        }
    }

    /// Add vectors to the index.
    ///
    /// Returns the slot ids (length == number of input vectors) so
    /// callers can key external metadata on the exact slot that was
    /// written. Slots freed by `delete()` are reused in LIFO order;
    /// the remainder is appended.
    fn add(&mut self, vectors: PyReadonlyArray2<f32>) -> Vec<i64> {
        let arr = vectors.as_array();
        let slice = arr.as_slice().expect("vectors must be contiguous");
        self.inner.add(slice)
    }

    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>) {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let slice = arr.as_slice().expect("queries must be contiguous");
        let results = self.inner.search(slice, k);

        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .unwrap()
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .unwrap()
            .into_pyarray(py);

        (scores, indices)
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner.write(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("{}", e))
        })
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbovec_core::TurboQuantIndex::load(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("{}", e))
        })?;
        Ok(Self { inner })
    }

    /// Warm up the search caches (rotation matrix, Lloyd-Max centroids,
    /// SIMD-blocked code layout) so the first `search` call does not pay
    /// the one-time initialisation cost.
    fn prepare(&self) {
        self.inner.prepare();
    }

    /// Tombstone one or more vector ids. Deleted ids are excluded from
    /// subsequent search results and their slots become available for
    /// reuse by the next `add`. Ids outside `[0, capacity())` and
    /// already-deleted ids are silently ignored.
    fn delete(&mut self, ids: Vec<i64>) {
        self.inner.delete(&ids);
    }

    fn is_deleted(&self, id: i64) -> bool {
        self.inner.is_deleted(id)
    }

    fn num_deleted(&self) -> usize {
        self.inner.num_deleted()
    }

    /// Physical slot count — includes tombstoned slots. Use `len()`
    /// for the live count.
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }
}

#[pymodule]
fn _turbovec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TurboQuantIndex>()?;
    Ok(())
}
