//! Sharding strategies for cluster document distribution.
//! Ported from splatdb Python.

use crate::cluster::health::GeoLocation;

/// Deterministic sharding by document ID hash.
/// Returns the shard index (0-based) for the given doc_id.
pub fn shard_by_hash(doc_id: &str, num_edges: usize) -> usize {
    if num_edges == 0 {
        return 0;
    }
    // Simple FNV-1a hash
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in doc_id.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash as usize) % num_edges
}

/// Shard by nearest centroid to embedding.
/// Routes document to edge with nearest centroid vector.
pub fn shard_by_cluster(embedding: &[f32], centroids: &[Vec<f32>], edge_ids: &[String]) -> Option<String> {
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
pub fn shard_by_geo(doc_lat: f64, doc_lon: f64, edge_locations: &[(String, GeoLocation)]) -> String {
    if edge_locations.is_empty() {
        return "edge-default".to_string();
    }

    let doc_loc = GeoLocation { latitude: doc_lat, longitude: doc_lon };
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
    fn test_shard_by_cluster() {
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let edges = vec!["e1".to_string(), "e2".to_string()];
        let result = shard_by_cluster(&[0.9, 0.1], &centroids, &edges);
        assert_eq!(result, Some("e1".to_string()));
    }

    #[test]
    fn test_haversine() {
        let loc1 = GeoLocation { latitude: 0.0, longitude: 0.0 };
        let loc2 = GeoLocation { latitude: 0.0, longitude: 1.0 };
        let dist = haversine_distance(&loc1, &loc2);
        assert!(dist > 0.0 && dist < 200.0); // ~111km
    }
}
