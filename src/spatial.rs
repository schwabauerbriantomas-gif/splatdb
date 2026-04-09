//! Spatial Memory Architecture — Wings / Rooms / Halls / Tunnels.
//!
//! Inspired by MemPalace: organize vector memory like physical space.
//!
//! - **Wing**: Top-level scope (project, persona, domain). Maps to document metadata tag.
//! - **Room**: Semantic grouping within a wing. Maps to KMeans++ coarse cluster + label.
//! - **Hall**: Memory type filter (fact, decision, event, error). Maps to metadata category.
//! - **Tunnel**: Cross-wing connection when the same room appears in two wings.
//!   Maps to a GraphSplat edge with relation_type "TUNNEL".
//!
//! ## Query Flow
//! ```text
//! Query: "auth decisions from project-x"
//!   1. Filter by wing="project-x"  → ~10% of corpus
//!   2. Filter by room="auth"       → ~2% of corpus
//!   3. Filter by hall="decisions"  → ~0.5% of corpus
//!   4. Vector search within that subspace → high recall, minimal noise
//! ```

use std::collections::{HashMap, HashSet};

/// Named cluster label for a KMeans++ coarse cluster.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoomLabel {
    /// The cluster index from KMeans++ coarse.
    pub cluster_idx: usize,
    /// Human-readable label (e.g., "auth", "billing", "migration").
    pub label: String,
    /// Which wing this room belongs to.
    pub wing: String,
    /// Document count in this room.
    pub doc_count: usize,
}

/// A tunnel connecting the same room across two wings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Tunnel {
    /// Wing A name.
    pub wing_a: String,
    /// Wing B name.
    pub wing_b: String,
    /// The shared room label.
    pub room_label: String,
    /// GraphSplat edge ID (if materialized).
    pub edge_id: Option<usize>,
}

/// Spatial query filter — reduces search space before vector distance computation.
#[derive(Debug, Clone, Default)]
pub struct SpatialFilter {
    /// Filter by wing (project/persona).
    pub wing: Option<String>,
    /// Filter by room (cluster label).
    pub room: Option<String>,
    /// Filter by hall (memory type: fact, decision, event, error).
    pub hall: Option<String>,
}

/// Spatial Memory index — tracks Wings/Rooms/Halls/Tunnels.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SpatialIndex {
    /// wing → { room_label → RoomLabel }
    wings: HashMap<String, HashMap<String, RoomLabel>>,
    /// Tunnel connections between wings.
    tunnels: Vec<Tunnel>,
    /// doc_id → spatial metadata (wing, room, hall)
    doc_spatial: HashMap<String, SpatialMeta>,
}

/// Per-document spatial metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpatialMeta {
    pub wing: Option<String>,
    pub room: Option<String>,
    pub hall: Option<String>,
}

impl SpatialIndex {
    /// Create a new empty spatial index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a document's spatial metadata (wing, room, hall).
    pub fn register_doc(
        &mut self,
        doc_id: &str,
        wing: Option<&str>,
        room: Option<&str>,
        hall: Option<&str>,
    ) {
        let meta = SpatialMeta {
            wing: wing.map(|s| s.to_string()),
            room: room.map(|s| s.to_string()),
            hall: hall.map(|s| s.to_string()),
        };

        // Update room doc count
        if let (Some(w), Some(r)) = (&meta.wing, &meta.room) {
            let wing_entry = self.wings.entry(w.clone()).or_default();
            let room_entry = wing_entry.entry(r.clone()).or_insert_with(|| RoomLabel {
                cluster_idx: 0, // Will be set when clusters are assigned
                label: r.clone(),
                wing: w.clone(),
                doc_count: 0,
            });
            room_entry.doc_count += 1;
        }

        self.doc_spatial.insert(doc_id.to_string(), meta);
    }

    /// Assign a cluster index to a room label.
    pub fn set_room_cluster(&mut self, wing: &str, room: &str, cluster_idx: usize) {
        if let Some(wing_entry) = self.wings.get_mut(wing) {
            if let Some(room_entry) = wing_entry.get_mut(room) {
                room_entry.cluster_idx = cluster_idx;
            }
        }
    }

    /// Get a document's spatial metadata.
    pub fn get_doc_meta(&self, doc_id: &str) -> Option<&SpatialMeta> {
        self.doc_spatial.get(doc_id)
    }

    /// Filter document IDs by spatial query.
    /// Returns doc IDs that match all provided filters (AND logic).
    /// Pass None for a field to skip that filter.
    pub fn filter(&self, spatial_filter: &SpatialFilter) -> Vec<String> {
        self.doc_spatial
            .iter()
            .filter(|(_, meta)| {
                if let Some(ref wing) = spatial_filter.wing {
                    if meta.wing.as_ref() != Some(wing) {
                        return false;
                    }
                }
                if let Some(ref room) = spatial_filter.room {
                    if meta.room.as_ref() != Some(room) {
                        return false;
                    }
                }
                if let Some(ref hall) = spatial_filter.hall {
                    if meta.hall.as_ref() != Some(hall) {
                        return false;
                    }
                }
                true
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Detect and create tunnels between wings that share the same room label.
    /// Returns the number of new tunnels created.
    pub fn detect_tunnels(&mut self) -> usize {
        // room_label → set of wings that have it
        let mut room_wings: HashMap<String, HashSet<String>> = HashMap::new();
        for (wing_name, rooms) in &self.wings {
            for room_label in rooms.keys() {
                room_wings
                    .entry(room_label.clone())
                    .or_default()
                    .insert(wing_name.clone());
            }
        }

        let mut new_count = 0;
        let existing: HashSet<(String, String, String)> = self
            .tunnels
            .iter()
            .map(|t| {
                let mut pair = vec![t.wing_a.clone(), t.wing_b.clone()];
                pair.sort();
                (pair[0].clone(), pair[1].clone(), t.room_label.clone())
            })
            .collect();

        for (room_label, wings) in &room_wings {
            if wings.len() < 2 {
                continue;
            }
            let wing_vec: Vec<&String> = wings.iter().collect();
            for i in 0..wing_vec.len() {
                for j in (i + 1)..wing_vec.len() {
                    let mut pair = vec![wing_vec[i].clone(), wing_vec[j].clone()];
                    pair.sort();
                    let key = (pair[0].clone(), pair[1].clone(), room_label.clone());
                    if !existing.contains(&key) {
                        self.tunnels.push(Tunnel {
                            wing_a: pair[0].clone(),
                            wing_b: pair[1].clone(),
                            room_label: room_label.clone(),
                            edge_id: None,
                        });
                        new_count += 1;
                    }
                }
            }
        }
        new_count
    }

    /// Get all tunnels.
    pub fn tunnels(&self) -> &[Tunnel] {
        &self.tunnels
    }

    /// Get all rooms for a wing.
    pub fn rooms_for_wing(&self, wing: &str) -> Vec<&RoomLabel> {
        self.wings
            .get(wing)
            .map(|rooms| rooms.values().collect())
            .unwrap_or_default()
    }

    /// Get all wing names.
    pub fn wing_names(&self) -> Vec<&str> {
        self.wings.keys().map(|s| s.as_str()).collect()
    }

    /// Get all unique room labels across all wings.
    pub fn all_room_labels(&self) -> HashSet<&str> {
        self.wings
            .values()
            .flat_map(|rooms| rooms.keys())
            .map(|s| s.as_str())
            .collect()
    }

    /// Get all unique hall values across all documents.
    pub fn all_hall_values(&self) -> HashSet<&str> {
        self.doc_spatial
            .values()
            .filter_map(|m| m.hall.as_deref())
            .collect()
    }

    /// Count total documents with spatial metadata.
    pub fn doc_count(&self) -> usize {
        self.doc_spatial.len()
    }

    /// Remove a document's spatial metadata.
    pub fn remove_doc(&mut self, doc_id: &str) {
        if let Some(meta) = self.doc_spatial.remove(doc_id) {
            // Decrement room doc count
            if let (Some(w), Some(r)) = (&meta.wing, &meta.room) {
                if let Some(wing_entry) = self.wings.get_mut(w) {
                    if let Some(room_entry) = wing_entry.get_mut(r) {
                        room_entry.doc_count = room_entry.doc_count.saturating_sub(1);
                    }
                }
            }
        }
    }

    /// Find tunnels that connect to a specific wing and room.
    pub fn tunnels_for(&self, wing: &str, room: &str) -> Vec<&Tunnel> {
        self.tunnels
            .iter()
            .filter(|t| {
                t.room_label == room
                    && (t.wing_a == wing || t.wing_b == wing)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_register_and_filter() {
        let mut idx = SpatialIndex::new();

        idx.register_doc("doc1", Some("project-x"), Some("auth"), Some("decision"));
        idx.register_doc("doc2", Some("project-x"), Some("auth"), Some("error"));
        idx.register_doc("doc3", Some("project-x"), Some("billing"), Some("fact"));
        idx.register_doc("doc4", Some("project-y"), Some("auth"), Some("decision"));

        // Filter by wing only
        let result = idx.filter(&SpatialFilter {
            wing: Some("project-x".into()),
            room: None,
            hall: None,
        });
        assert_eq!(result.len(), 3);

        // Filter by wing + room
        let result = idx.filter(&SpatialFilter {
            wing: Some("project-x".into()),
            room: Some("auth".into()),
            hall: None,
        });
        assert_eq!(result.len(), 2);

        // Filter by wing + room + hall
        let result = idx.filter(&SpatialFilter {
            wing: Some("project-x".into()),
            room: Some("auth".into()),
            hall: Some("decision".into()),
        });
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "doc1");
    }

    #[test]
    fn test_tunnel_detection() {
        let mut idx = SpatialIndex::new();

        idx.register_doc("d1", Some("wing-a"), Some("auth"), Some("fact"));
        idx.register_doc("d2", Some("wing-b"), Some("auth"), Some("fact"));
        idx.register_doc("d3", Some("wing-c"), Some("billing"), Some("fact"));

        let new = idx.detect_tunnels();
        assert_eq!(new, 1); // Only "auth" connects wing-a and wing-b
        assert_eq!(idx.tunnels().len(), 1);
        assert_eq!(idx.tunnels()[0].room_label, "auth");
    }

    #[test]
    fn test_no_duplicate_tunnels() {
        let mut idx = SpatialIndex::new();

        idx.register_doc("d1", Some("a"), Some("auth"), None);
        idx.register_doc("d2", Some("b"), Some("auth"), None);

        assert_eq!(idx.detect_tunnels(), 1);
        assert_eq!(idx.detect_tunnels(), 0); // No new tunnels
    }

    #[test]
    fn test_remove_doc() {
        let mut idx = SpatialIndex::new();
        idx.register_doc("d1", Some("w"), Some("r"), None);
        assert_eq!(idx.doc_count(), 1);
        idx.remove_doc("d1");
        assert_eq!(idx.doc_count(), 0);
    }
}
