//! Sharding strategies for cluster document distribution.
//! Ported from splatsdb Python.

use crate::cluster::health::GeoLocation;

/// FNV-1a-64 with a splitmix64 finalizer.
///
/// Deterministic and stable across builds and platforms (unlike
/// `DefaultHasher`). The finalizer matters: raw FNV-1a correlates hashes of
/// short strings that differ only in their last byte, which made one edge
/// systematically outrank another in rendezvous weighting.
fn stable_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // splitmix64 finalizer
    hash = hash.wrapping_add(0x9E37_79B9_7F4A_7C15);
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    hash ^ (hash >> 31)
}

/// Rendezvous (highest-random-weight) sharding over explicit edge IDs.
///
/// For each edge, weight = stable_hash(doc_id + "|" + edge_id); the edge with
/// the maximum weight wins. Unlike `hash % n`, the winner is independent of
/// edge ORDER, and adding/removing an edge only remaps ~1/N of documents.
/// Returns the winning edge's index in `edge_ids`, or None when empty.
pub fn shard_by_rendezvous(doc_id: &str, edge_ids: &[String]) -> Option<usize> {
    if edge_ids.is_empty() {
        return None;
    }
    let mut best_idx = 0;
    let mut best_weight = stable_hash(format!("{}|{}", doc_id, edge_ids[0]).as_bytes());
    for (i, edge_id) in edge_ids.iter().enumerate().skip(1) {
        let w = stable_hash(format!("{}|{}", doc_id, edge_id).as_bytes());
        if w > best_weight {
            best_weight = w;
            best_idx = i;
        }
    }
    Some(best_idx)
}

/// Deterministic sharding by document ID.
///
/// Rendezvous hashing over the first `num_edges` synthetic edge ids
/// ("0".."n-1"): growing the edge set only remaps a minority of documents,
/// whereas the previous `hash % num_edges` remapped every document whenever
/// the topology changed. Returns the shard index (0-based) for the doc_id.
pub fn shard_by_hash(doc_id: &str, num_edges: usize) -> usize {
    if num_edges == 0 {
        return 0;
    }
    let synthetic: Vec<String> = (0..num_edges).map(|i| i.to_string()).collect();
    shard_by_rendezvous(doc_id, &synthetic).unwrap_or(0)
}

/// Shard by nearest centroid to embedding.
/// Routes document to edge with nearest centroid vector.
pub fn shard_by_cluster(
    embedding: &[f32],
    centroids: &[Vec<f32>],
    edge_ids: &[String],
) -> Option<String> {
    if centroids.is_empty() || edge_ids.is_empty() || embedding.is_empty() {
        return None;
    }

    let mut best_idx = 0;
    let mut best_dist = f64::MAX;

    for (i, centroid) in centroids.iter().enumerate() {
        let dist = euclidean_distance(embedding, centroid);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    let idx = best_idx % edge_ids.len();
    Some(edge_ids[idx].clone())
}

/// Shard by geographic proximity.
/// Routes document to nearest edge by haversine distance.
pub fn shard_by_geo(
    doc_lat: f64,
    doc_lon: f64,
    edge_locations: &[(String, GeoLocation)],
) -> String {
    if edge_locations.is_empty() {
        return "edge-default".to_string();
    }

    let doc_loc = GeoLocation {
        latitude: doc_lat,
        longitude: doc_lon,
    };
    let mut nearest = &edge_locations[0].0;
    let mut min_dist = f64::MAX;

    for (edge_id, loc) in edge_locations {
        let d = haversine_distance(&doc_loc, loc);
        if d < min_dist {
            min_dist = d;
            nearest = edge_id;
        }
    }

    nearest.clone()
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    let sum: f64 = (0..len)
        .map(|i| {
            let d = a[i] as f64 - b[i] as f64;
            d * d
        })
        .sum();
    sum.sqrt()
}

/// Haversine distance in kilometers between two geographic points.
fn haversine_distance(loc1: &GeoLocation, loc2: &GeoLocation) -> f64 {
    let lat1 = loc1.latitude.to_radians();
    let lat2 = loc2.latitude.to_radians();
    let dlat = (loc2.latitude - loc1.latitude).to_radians();
    let dlon = (loc2.longitude - loc1.longitude).to_radians();

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    c * 6371.0 // Earth radius in km
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_by_hash_deterministic() {
        let idx1 = shard_by_hash("doc_123", 4);
        let idx2 = shard_by_hash("doc_123", 4);
        assert_eq!(idx1, idx2);
        assert!(idx1 < 4);
    }

    #[test]
    fn test_shard_by_hash_distribution() {
        let n = 4;
        let mut counts = vec![0usize; n];
        for i in 0..100 {
            let idx = shard_by_hash(&format!("doc_{}", i), n);
            counts[idx] += 1;
        }
        // All shards should get some items
        assert!(counts.iter().all(|&c| c > 0));
    }

    #[test]
    fn test_rendezvous_independent_of_edge_order() {
        let edges = vec!["e1".to_string(), "e2".to_string(), "e3".to_string()];
        let mut reversed = edges.clone();
        reversed.reverse();
        for i in 0..50 {
            let doc = format!("doc_{}", i);
            let winner_a = &edges[shard_by_rendezvous(&doc, &edges).unwrap()];
            let winner_b = &reversed[shard_by_rendezvous(&doc, &reversed).unwrap()];
            assert_eq!(
                winner_a, winner_b,
                "same doc + same edge set must pick the same edge regardless of order"
            );
        }
    }

    #[test]
    fn test_rendezvous_minimal_disruption_on_topology_change() {
        let e3 = vec!["e1".to_string(), "e2".to_string(), "e3".to_string()];
        let e4 = vec![
            "e1".to_string(),
            "e2".to_string(),
            "e3".to_string(),
            "e4".to_string(),
        ];
        let total = 200;
        let mut moved = 0;
        for i in 0..total {
            let doc = format!("doc_{}", i);
            let before = &e3[shard_by_rendezvous(&doc, &e3).unwrap()];
            let after = &e4[shard_by_rendezvous(&doc, &e4).unwrap()];
            if before != after {
                moved += 1;
            }
        }
        // Ideal is ~total/4 (only docs where the new edge wins the tournament).
        // The old `hash % n` remapped EVERY doc when the edge count changed.
        assert!(moved > 0, "the new edge should win some docs");
        assert!(
            moved < total / 2,
            "only a minority should move on topology change: {}/{}",
            moved,
            total
        );
    }

    #[test]
    fn test_shard_by_cluster() {
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let edges = vec!["e1".to_string(), "e2".to_string()];
        let result = shard_by_cluster(&[0.9, 0.1], &centroids, &edges);
        assert_eq!(result, Some("e1".to_string()));
    }

    #[test]
    fn test_haversine() {
        let loc1 = GeoLocation {
            latitude: 0.0,
            longitude: 0.0,
        };
        let loc2 = GeoLocation {
            latitude: 0.0,
            longitude: 1.0,
        };
        let dist = haversine_distance(&loc1, &loc2);
        assert!(dist > 0.0 && dist < 200.0); // ~111km
    }
}
