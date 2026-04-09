//! CLI handlers for spatial memory commands.

use splatdb::spatial::{SpatialFilter, SpatialIndex};

/// Search with spatial memory filters (Wing/Room/Hall).
///
/// This is a placeholder that demonstrates the spatial query API.
/// Full integration requires wiring into the search pipeline to
/// pre-filter documents before vector search.
pub fn cmd_spatial_search(
    query: String,
    wing: Option<String>,
    room: Option<String>,
    hall: Option<String>,
    k: usize,
) {
    let filter = SpatialFilter { wing, room, hall };

    println!("Spatial Search");
    println!("──────────────");
    println!("Query:  {}", query);
    if let Some(ref w) = filter.wing {
        println!("Wing:   {}", w);
    }
    if let Some(ref r) = filter.room {
        println!("Room:   {}", r);
    }
    if let Some(ref h) = filter.hall {
        println!("Hall:   {}", h);
    }
    println!("K:      {}", k);
    println!();
    println!("Note: Spatial search requires documents indexed with spatial metadata.");
    println!("      Use doc-add with metadata {{\"wing\": \"...\", \"room\": \"...\", \"hall\": \"...\"}}");
    println!("      then spatial-search will filter before vector search.");
}

/// Show spatial memory structure (wings, rooms, tunnels).
pub fn cmd_spatial_info() {
    let index = SpatialIndex::new();

    println!("Spatial Memory Structure");
    println!("════════════════════════");

    let wings = index.wing_names();
    if wings.is_empty() {
        println!("\n  No spatial data indexed yet.");
        println!("  Index documents with wing/room/hall metadata to populate.");
        println!();
        println!("  Example:");
        println!("    splatdb doc-add --id d1 --text \"auth decision\" \\");
        println!("      --metadata '{{\"wing\": \"project-x\", \"room\": \"auth\", \"hall\": \"decision\"}}'");
    } else {
        println!("\n  Wings: {}", wings.join(", "));
        for wing in wings {
            let rooms = index.rooms_for_wing(wing);
            println!("\n  Wing: {}", wing);
            for room in rooms {
                println!("    Room: {} (cluster {}, {} docs)", room.label, room.cluster_idx, room.doc_count);
            }
        }

        let tunnels = index.tunnels();
        if !tunnels.is_empty() {
            println!("\n  Tunnels:");
            for t in tunnels {
                println!("    {} <-> {} via room \"{}\"", t.wing_a, t.wing_b, t.room_label);
            }
        }
    }

    println!();
    println!("  Architecture:");
    println!("    Wing  = Project / Persona / Domain (metadata tag)");
    println!("    Room  = Semantic cluster (KMeans++ + label)");
    println!("    Hall  = Memory type (fact, decision, event, error)");
    println!("    Tunnel = Cross-wing connection (auto-detected)");
}
