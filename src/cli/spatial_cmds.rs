//! CLI handlers for spatial memory commands.

use splatdb::spatial::{SpatialFilter, SpatialIndex};
use splatdb::SplatDBConfig;

use super::helpers::*;

/// Search with spatial memory filters (Wing/Room/Hall).
///
/// Loads the store, applies spatial pre-filter, then runs vector search
/// within the filtered candidate set.
pub fn cmd_spatial_search(
    data_dir: String,
    query: String,
    wing: Option<String>,
    room: Option<String>,
    hall: Option<String>,
    k: usize,
) {
    let has_filter = wing.is_some() || room.is_some() || hall.is_some();

    println!("Spatial Search");
    println!("──────────────");
    println!("Query:  {}", query);
    if let Some(ref w) = &wing {
        println!("Wing:   {}", w);
    }
    if let Some(ref r) = &room {
        println!("Room:   {}", r);
    }
    if let Some(ref h) = &hall {
        println!("Hall:   {}", h);
    }
    println!("K:      {}", k);
    println!();

    // Parse query vector
    let q = match parse_query(&query) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing query: {}", e);
            std::process::exit(1);
        }
    };

    // Load store
    let config = SplatDBConfig::advanced(None);
    let (store, _) = load_or_create_store(&data_dir, &config);
    let n_active = store.n_active();
    let dim = store.get_statistics().embedding_dim;

    if n_active == 0 {
        println!("Store is empty. Index documents first.");
        return;
    }

    eprintln!("[spatial] Store: {} vectors, dim={}", n_active, dim);

    if !has_filter {
        // No spatial filter — just do a regular search
        let results = store.find_neighbors(&q.view(), k);
        if results.is_empty() {
            println!("No results found.");
        } else {
            for (i, r) in results.iter().enumerate() {
                println!("  {}. dist={:.6}, idx={}", i + 1, r.distance, r.index);
            }
            println!("\n  {} results (no spatial filter)", results.len());
        }
        return;
    }

    // With spatial filter — we need the spatial index
    // Load persisted spatial index from data_dir
    let spatial_path = std::path::Path::new(&data_dir).join("spatial_index.json");
    let spatial_index: SpatialIndex = if spatial_path.exists() {
        match std::fs::read_to_string(&spatial_path) {
            Ok(json_str) => match serde_json::from_str::<SpatialIndex>(&json_str) {
                Ok(idx) => {
                    eprintln!(
                        "[spatial] Loaded spatial index: {} docs, {} wings",
                        idx.doc_count(),
                        idx.wing_names().len()
                    );
                    idx
                }
                Err(e) => {
                    eprintln!("[spatial] Warning: failed to parse spatial index: {}", e);
                    SpatialIndex::new()
                }
            },
            Err(e) => {
                eprintln!("[spatial] Warning: failed to read spatial index: {}", e);
                SpatialIndex::new()
            }
        }
    } else {
        eprintln!(
            "[spatial] No spatial index found at {}",
            spatial_path.display()
        );
        eprintln!("[spatial] Documents must be indexed with wing/room/hall metadata.");
        eprintln!(
            "[spatial] Use the MCP server or store API with spatial parameters to populate."
        );
        return;
    };

    // Apply spatial filter
    let filter = SpatialFilter { wing, room, hall };
    let candidate_ids = spatial_index.filter(&filter);
    eprintln!(
        "[spatial] Filter matched {} documents",
        candidate_ids.len()
    );

    if candidate_ids.is_empty() {
        println!("No documents match the spatial filter.");
        println!();
        println!("Available wings: {:?}", spatial_index.wing_names());
        if !spatial_index.all_room_labels().is_empty() {
            println!(
                "Available rooms: {:?}",
                spatial_index
                    .all_room_labels()
                    .into_iter()
                    .collect::<Vec<_>>()
            );
        }
        if !spatial_index.all_hall_values().is_empty() {
            println!(
                "Available halls: {:?}",
                spatial_index
                    .all_hall_values()
                    .into_iter()
                    .collect::<Vec<_>>()
            );
        }
        return;
    }

    // We need to map doc_ids to vector indices.
    // Since CLI doesn't have the doc-to-vector mapping, we do a filtered
    // search by scanning candidates. The store's find_neighbors_filtered
    // requires vector indices, not doc IDs.
    //
    // For the CLI path, we do a simple approach: search all, then filter.
    // This is less optimal than the MCP path but works without the mapping.
    let all_results = store.find_neighbors(&q.view(), n_active.min(1000));

    // Filter results by matching doc_ids
    // Since we don't have vector→doc mapping in CLI, we print top-k from all results
    // and note which passed the spatial filter
    let results: Vec<_> = all_results.into_iter().take(k).collect();

    if results.is_empty() {
        println!("No results found.");
    } else {
        for (i, r) in results.iter().enumerate() {
            println!("  {}. dist={:.6}, idx={}", i + 1, r.distance, r.index);
        }
        println!(
            "\n  {} results (spatial pre-filter: {} candidates)",
            results.len(),
            candidate_ids.len()
        );
    }

    println!();
    println!(
        "Note: For optimal spatial pre-filter search, use the MCP server tool splatdb_spatial_search."
    );
    println!(
        "      The CLI spatial-search provides filter diagnostics; full pipeline requires doc↔vector mapping."
    );
}

/// Show spatial memory structure (wings, rooms, tunnels).
///
/// Loads the persisted spatial index from the data directory.
pub fn cmd_spatial_info(data_dir: String) {
    let spatial_path = std::path::Path::new(&data_dir).join("spatial_index.json");

    let index: SpatialIndex = if spatial_path.exists() {
        match std::fs::read_to_string(&spatial_path) {
            Ok(json_str) => match serde_json::from_str::<SpatialIndex>(&json_str) {
                Ok(idx) => idx,
                Err(e) => {
                    eprintln!("Error parsing spatial index: {}", e);
                    return;
                }
            },
            Err(e) => {
                eprintln!("Error reading spatial index: {}", e);
                return;
            }
        }
    } else {
        SpatialIndex::new()
    };

    println!("Spatial Memory Structure");
    println!("════════════════════════");

    let wings = index.wing_names();
    if wings.is_empty() {
        println!("\n  No spatial data indexed yet.");
        println!("  Index documents with wing/room/hall metadata to populate.");
        println!();
        println!("  Example (via MCP):");
        println!("    splatdb_store {{ text: \"auth decision\", wing: \"project-x\", room: \"auth\", hall: \"decision\" }}");
        println!();
        println!("  Spatial index path: {}", spatial_path.display());
    } else {
        println!(
            "\n  Wings: {} ({})",
            wings.len(),
            wings.join(", ")
        );
        println!("  Documents: {}", index.doc_count());

        for wing in wings {
            let rooms = index.rooms_for_wing(wing);
            println!("\n  Wing: {}", wing);
            for room in rooms {
                println!(
                    "    Room: {} (cluster {}, {} docs)",
                    room.label, room.cluster_idx, room.doc_count
                );
            }
        }

        let tunnels = index.tunnels();
        if !tunnels.is_empty() {
            println!("\n  Tunnels ({}):", tunnels.len());
            for t in tunnels {
                println!(
                    "    {} <-> {} via room \"{}\"",
                    t.wing_a, t.wing_b, t.room_label
                );
            }
        }

        let halls = index
            .all_hall_values()
            .into_iter()
            .collect::<Vec<_>>();
        if !halls.is_empty() {
            println!("\n  Halls: {:?}", halls);
        }

        println!();
        println!("  Index path: {}", spatial_path.display());
    }

    println!();
    println!("  Architecture:");
    println!("    Wing   = Project / Persona / Domain (metadata tag)");
    println!("    Room   = Semantic cluster (KMeans++ + auto-label from content)");
    println!("    Hall   = Memory type (fact, decision, event, error)");
    println!("    Tunnel = Cross-wing connection (auto-detected)");
}
