//! Integration tests for tombstone-based deletion on `TurboQuantIndex`.

use turbovec::TurboQuantIndex;

fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..(n * dim) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 32) as u32) as f32 / u32::MAX as f32;
        out.push(u * 2.0 - 1.0);
    }
    out
}

fn build_index(n: usize) -> TurboQuantIndex {
    let dim = 128;
    let bit_width = 4;
    let vectors = make_vectors(n, dim, 1);
    let mut idx = TurboQuantIndex::new(dim, bit_width);
    idx.add(&vectors);
    idx
}

#[test]
fn delete_marks_and_len_reflects_live_count() {
    let mut idx = build_index(64);
    assert_eq!(idx.len(), 64);
    assert_eq!(idx.capacity(), 64);
    assert_eq!(idx.num_deleted(), 0);

    idx.delete(&[3, 5, 7]);

    assert_eq!(idx.len(), 61);
    assert_eq!(idx.capacity(), 64);
    assert_eq!(idx.num_deleted(), 3);
    assert!(idx.is_deleted(3));
    assert!(idx.is_deleted(5));
    assert!(!idx.is_deleted(4));
}

#[test]
fn delete_is_idempotent_and_bounds_safe() {
    let mut idx = build_index(16);
    idx.delete(&[0, 0, 0]);
    assert_eq!(idx.num_deleted(), 1);
    idx.delete(&[-1, 99, 17]); // all out-of-range, silently ignored
    assert_eq!(idx.num_deleted(), 1);
}

#[test]
fn search_excludes_deleted_ids() {
    let mut idx = build_index(128);
    idx.prepare();

    let queries = make_vectors(1, idx.dim(), 7);
    let top = idx.search(&queries, 5);
    let all_ids: Vec<i64> = top.indices_for_query(0).to_vec();
    assert_eq!(all_ids.len(), 5);

    // Delete the top hit and re-query; it must not appear.
    let victim = all_ids[0];
    idx.delete(&[victim]);
    let top2 = idx.search(&queries, 5);
    let after: Vec<i64> = top2.indices_for_query(0).to_vec();
    assert!(!after.contains(&victim), "deleted id came back: {:?}", after);
    assert_eq!(after.len(), 5);
}

#[test]
fn add_reuses_deleted_slots() {
    let mut idx = build_index(8);
    idx.delete(&[2, 4]);
    assert_eq!(idx.len(), 6);
    assert_eq!(idx.capacity(), 8);
    assert_eq!(idx.num_deleted(), 2);

    // Add 3 more vectors: 2 should reuse slots 4 and 2 (LIFO), 1 appends.
    let extra = make_vectors(3, idx.dim(), 99);
    idx.add(&extra);

    assert_eq!(idx.len(), 9);
    assert_eq!(idx.capacity(), 9);
    assert_eq!(idx.num_deleted(), 0);
    assert!(!idx.is_deleted(2));
    assert!(!idx.is_deleted(4));
}

#[test]
fn search_still_works_when_everything_is_deleted() {
    let mut idx = build_index(8);
    let ids: Vec<i64> = (0..8).collect();
    idx.delete(&ids);
    assert_eq!(idx.len(), 0);

    let queries = make_vectors(1, idx.dim(), 1);
    let top = idx.search(&queries, 4);
    // k clamps to live = 0.
    assert_eq!(top.k, 0);
}

#[test]
fn write_load_roundtrips_tombstones() {
    let tmp = std::env::temp_dir().join("tv_delete_roundtrip.tv");
    let _ = std::fs::remove_file(&tmp);

    let mut idx = build_index(16);
    idx.delete(&[1, 4, 9, 15]);
    idx.write(&tmp).unwrap();

    let loaded = TurboQuantIndex::load(&tmp).unwrap();
    assert_eq!(loaded.len(), 12);
    assert_eq!(loaded.capacity(), 16);
    assert_eq!(loaded.num_deleted(), 4);
    for &d in &[1i64, 4, 9, 15] {
        assert!(loaded.is_deleted(d));
    }

    // Confirm search on the loaded index excludes deletes.
    let queries = make_vectors(2, loaded.dim(), 3);
    let top = loaded.search(&queries, 12);
    for qi in 0..top.nq {
        for &id in top.indices_for_query(qi) {
            assert!(id < 0 || !loaded.is_deleted(id));
        }
    }

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn legacy_file_without_tombstone_section_still_loads() {
    // When no slot is tombstoned, write() emits the original v1 layout
    // (no section tag). Loading that back must yield an all-zero
    // tombstone vector — no spurious dead slots.
    let tmp = std::env::temp_dir().join("tv_legacy_layout.tv");
    let _ = std::fs::remove_file(&tmp);

    let idx = build_index(10);
    idx.write(&tmp).unwrap();
    let loaded = TurboQuantIndex::load(&tmp).unwrap();
    assert_eq!(loaded.len(), 10);
    assert_eq!(loaded.num_deleted(), 0);

    let _ = std::fs::remove_file(&tmp);
}
