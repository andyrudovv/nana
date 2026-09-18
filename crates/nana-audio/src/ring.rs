//! Fixed-capacity pre-roll ring holding the most recent audio.

use nana_core::audio::{seconds_to_samples, PREROLL_SECONDS};

/// Circular buffer addressed by absolute sample index.
///
/// Absolute indices let a consumer detect that it fell behind and its window was
/// overwritten, which a plain read cursor silently hides.
pub struct PreRoll {
    buf: Vec<i16>,
    written: u64,
}

impl PreRoll {
    pub fn new(capacity_samples: usize) -> Self {
        assert!(capacity_samples > 0, "pre-roll capacity must be non-zero");
        Self { buf: vec![0; capacity_samples], written: 0 }
    }

    pub fn with_default_capacity() -> Self {
        Self::new(seconds_to_samples(PREROLL_SECONDS))
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Total samples ever written; also the absolute index of the next write.
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Absolute index of the oldest sample still held.
    pub fn oldest(&self) -> u64 {
        self.written.saturating_sub(self.buf.len() as u64)
    }

    pub fn push(&mut self, samples: &[i16]) {
        let cap = self.buf.len();
        // A burst larger than the ring only leaves its tail behind, but the
        // dropped prefix still advances both the write position and the
        // absolute index — otherwise later reads land on the wrong samples.
        let dropped = samples.len().saturating_sub(cap);
        let kept = &samples[dropped..];

        let start = ((self.written + dropped as u64) % cap as u64) as usize;
        let first = (cap - start).min(kept.len());
        self.buf[start..start + first].copy_from_slice(&kept[..first]);
        self.buf[..kept.len() - first].copy_from_slice(&kept[first..]);
        self.written += samples.len() as u64;
    }

    /// The most recent `count` samples, oldest first. Shorter when the ring has
    /// not filled up yet.
    pub fn tail(&self, count: usize) -> Vec<i16> {
        let count = count.min(self.buf.len()).min(self.written as usize);
        self.range(self.written - count as u64, count).unwrap_or_default()
    }

    /// `count` samples starting at an absolute index, or `None` when that window
    /// has already been overwritten or not yet been written.
    pub fn range(&self, from: u64, count: usize) -> Option<Vec<i16>> {
        if from < self.oldest() || from + count as u64 > self.written {
            return None;
        }
        let cap = self.buf.len();
        let start = (from % cap as u64) as usize;
        let mut out = Vec::with_capacity(count);
        let first = (cap - start).min(count);
        out.extend_from_slice(&self.buf[start..start + first]);
        out.extend_from_slice(&self.buf[..count - first]);
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_returns_most_recent_samples() {
        let mut ring = PreRoll::new(8);
        ring.push(&[1, 2, 3, 4, 5]);
        assert_eq!(ring.tail(3), vec![3, 4, 5]);
        assert_eq!(ring.tail(99), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn wraps_and_drops_oldest() {
        let mut ring = PreRoll::new(4);
        ring.push(&[1, 2, 3]);
        ring.push(&[4, 5]);
        assert_eq!(ring.written(), 5);
        assert_eq!(ring.oldest(), 1);
        assert_eq!(ring.tail(4), vec![2, 3, 4, 5]);
    }

    #[test]
    fn range_reports_overwritten_windows() {
        let mut ring = PreRoll::new(4);
        ring.push(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(ring.range(0, 2), None, "window fell out of the ring");
        assert_eq!(ring.range(2, 4), Some(vec![3, 4, 5, 6]));
        assert_eq!(ring.range(4, 4), None, "window runs past what was written");
    }

    #[test]
    fn push_larger_than_capacity_keeps_the_tail() {
        let mut ring = PreRoll::new(3);
        ring.push(&[1, 2, 3, 4, 5]);
        assert_eq!(ring.tail(3), vec![3, 4, 5]);
        assert_eq!(ring.written(), 5);
    }
}
