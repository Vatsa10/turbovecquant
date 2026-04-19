//! Read/write TurboVec index files (.tv format).
//!
//! Binary layout:
//!   Header (9 bytes): bit_width (u8) | dim (u32 LE) | n_vectors (u32 LE)
//!   Packed codes:     (dim / 8) * bit_width * n_vectors bytes
//!   Norms:            n_vectors * f32 LE
//!   [optional] Tombstone section — present only when any slot is deleted:
//!     Tag (1 byte):    0x54 ('T')
//!     Bitmap:          ceil(n_vectors / 8) bytes, LSB = slot 0, 1 = deleted
//!
//! Files written before the tombstone section existed remain readable —
//! load() treats a missing section as "no deletions".

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const HEADER_SIZE: usize = 9;
const TOMBSTONE_TAG: u8 = 0x54;

pub fn write(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    norms: &[f32],
    tombstones: &[u8],
) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);

    // Header
    f.write_all(&[bit_width as u8])?;
    f.write_all(&(dim as u32).to_le_bytes())?;
    f.write_all(&(n_vectors as u32).to_le_bytes())?;

    // Packed codes
    f.write_all(packed_codes)?;

    // Norms as raw f32 bytes
    for &n in norms {
        f.write_all(&n.to_le_bytes())?;
    }

    // Optional tombstone section — only written when at least one slot
    // is tombstoned. Keeping files written with no deletions byte-for-
    // byte identical to the legacy format protects backward-readable
    // behaviour for third-party tooling.
    if tombstones.iter().any(|&b| b != 0) {
        let bitmap_len = n_vectors.div_ceil(8);
        let mut bitmap = vec![0u8; bitmap_len];
        for (i, &t) in tombstones.iter().enumerate() {
            if t != 0 {
                bitmap[i >> 3] |= 1 << (i & 7);
            }
        }
        f.write_all(&[TOMBSTONE_TAG])?;
        f.write_all(&bitmap)?;
    }

    f.flush()?;
    Ok(())
}

pub fn load(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>, Vec<u8>)> {
    let mut f = BufReader::new(File::open(path)?);

    // Header
    let mut header = [0u8; HEADER_SIZE];
    f.read_exact(&mut header)?;

    let bit_width = header[0] as usize;
    let dim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let n_vectors = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;

    // Packed codes
    let packed_bytes = (dim / 8) * bit_width * n_vectors;
    let mut packed_codes = vec![0u8; packed_bytes];
    f.read_exact(&mut packed_codes)?;

    // Norms
    let mut norms_bytes = vec![0u8; n_vectors * 4];
    f.read_exact(&mut norms_bytes)?;
    let norms: Vec<f32> = norms_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Optional tombstone section. Absence is normal (legacy file or no
    // deletions). A single-byte read that returns EOF means no section.
    let mut tag = [0u8; 1];
    let tombstones = match f.read(&mut tag)? {
        0 => vec![0u8; n_vectors],
        _ if tag[0] == TOMBSTONE_TAG => {
            let bitmap_len = n_vectors.div_ceil(8);
            let mut bitmap = vec![0u8; bitmap_len];
            f.read_exact(&mut bitmap)?;
            let mut ts = vec![0u8; n_vectors];
            for i in 0..n_vectors {
                if bitmap[i >> 3] & (1 << (i & 7)) != 0 {
                    ts[i] = 1;
                }
            }
            ts
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown section tag 0x{:02x}", tag[0]),
            ));
        }
    };

    Ok((bit_width, dim, n_vectors, packed_codes, norms, tombstones))
}
