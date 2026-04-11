//! CLI command handlers — extracted from main.rs

mod bench_longmemeval;
mod cluster_cmds;
mod data_cmds;
mod graph_cmds;
mod helpers;
mod index_cmds;
mod ml_cmds;
mod search_cmds;
mod spatial_cmds;
mod verbatim_cmds;

use clap::Subcommand;
use std::path::PathBuf;

use crate::Cli;

#[derive(Subcommand)]
pub enum Commands {
    /// Add vectors from a binary file (rows: u64, cols: u64, f32 data)
    Index {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long, default_value = "default")]
        shard: String,
    },
    /// Append vectors to existing store with incremental HNSW insert
    Append {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long, default_value = "default")]
        shard: String,
    },
    /// Search for k nearest neighbors
    Search {
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value = "10")]
        k: usize,
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Search using a query vector from a binary file
    SearchFile {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long, default_value = "10")]
        k: usize,
        #[arg(short = 'f', long, default_value = "json")]
        format: String,
    },
    /// Show store statistics
    Status {
        #[arg(short, long)]
        verbose: bool,
    },
    /// SOC: check system criticality
    SocCheck,
    /// SOC: trigger avalanche reorganization
    SocAvalanche {
        #[arg(short, long)]
        seed: Option<usize>,
    },
    /// SOC: relax the system toward lower energy
    SocRelax {
        #[arg(short, long, default_value = "10")]
        iterations: usize,
    },
    /// Save current state to disk
    Save {
        #[arg(short, long, default_value = "default")]
        shard: String,
    },
    /// Load state from disk
    Load {
        #[arg(short, long, default_value = "default")]
        shard: String,
    },
    /// List all stored shards
    List,
    /// Save document metadata
    DocAdd {
        #[arg(short, long)]
        id: String,
        #[arg(short, long)]
        text: String,
        #[arg(short, long)]
        metadata: Option<String>,
    },
    /// Get document metadata
    DocGet {
        #[arg(short, long)]
        id: String,
    },
    /// Delete a document (soft delete)
    DocDel {
        #[arg(short, long)]
        id: String,
    },
    /// Backup all data
    Backup {
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Quantize vectors with TurboQuant and search compressed codes
    QuantIndex {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long, default_value = "8")]
        bits: u8,
        #[arg(short, long, default_value = "turbo")]
        algorithm: String,
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },
    /// Search using quantized codes
    QuantSearch {
        #[arg(short, long)]
        query: String,
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
    },
    /// Show quantization statistics
    QuantStatus,
    /// Show GPU/CUDA status and info
    GpuInfo,
    /// Ingest dataset via DatasetTransformer
    Ingest {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        n_clusters: Option<usize>,
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },
    /// Ingest dataset with hierarchical 2-level KMeans
    IngestHierarchical {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short = 'n', long, default_value = "10")]
        n_clusters: usize,
        #[arg(short, long, default_value = "2")]
        min_cluster_size: usize,
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },
    /// Ingest via Leader Clustering — O(n) single-pass
    IngestLeader {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short = 'n', long, default_value = "100")]
        target_clusters: usize,
        #[arg(short = 't', long)]
        threshold: Option<f64>,
        #[arg(short, long, default_value = "42")]
        seed: u64,
    },
    /// Search using all enabled backends fused
    FusedSearch {
        #[arg(short, long)]
        query: Option<String>,
        #[arg(long)]
        query_file: Option<PathBuf>,
        #[arg(short, long, default_value = "10")]
        k: usize,
    },
    /// Show which subsystems each preset enables
    PresetInfo {
        #[arg(short, long)]
        preset: Option<String>,
    },
    /// Search using HNSW index only
    HnswSearch {
        #[arg(short, long)]
        query: Option<String>,
        #[arg(long)]
        query_file: Option<PathBuf>,
        #[arg(short, long, default_value = "10")]
        k: usize,
    },
    /// Search using LSH index only
    LshSearch {
        #[arg(short, long)]
        query: Option<String>,
        #[arg(long)]
        query_file: Option<PathBuf>,
        #[arg(short, long, default_value = "10")]
        k: usize,
    },
    /// Benchmark GPU + DatasetTransformer: ingest + search pipeline
    BenchGpuIngest {
        #[arg(short = 'n', long, default_value = "100000")]
        n_vectors: usize,
        #[arg(short = 'd', long, default_value = "640")]
        dim: usize,
        #[arg(short = 'k', long, default_value = "100")]
        n_clusters: usize,
        #[arg(short = 'q', long, default_value = "100")]
        n_queries: usize,
    },
    /// Benchmark GPU vs CPU search performance
    BenchGpu {
        #[arg(short = 'n', long, default_value = "10000")]
        n_vectors: usize,
        #[arg(short = 'd', long, default_value = "640")]
        dim: usize,
        #[arg(short = 'q', long, default_value = "100")]
        n_queries: usize,
        #[arg(short = 'k', long, default_value = "10")]
        top_k: usize,
        #[arg(short, long, default_value = "l2")]
        metric: String,
    },
    /// Start HTTP API server for Python bindings
    Serve {
        #[arg(short, long, default_value = "8199")]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    /// Start MCP server (stdio transport for AI agent integration)
    Mcp,
    // ── Graph Splat Commands ───────────────────────────────────────────
    /// Add a document node to the graph store
    GraphAddDoc {
        #[arg(short, long)]
        text: String,
        #[arg(short, long)]
        embedding: String,
    },
    /// Add an entity node to the graph store
    GraphAddEntity {
        #[arg(short, long)]
        name: String,
        #[arg(short, long)]
        embedding: String,
        #[arg(short = 't', long, default_value = "default")]
        entity_type: String,
    },
    /// Add a relation edge between two graph nodes
    GraphAddRelation {
        #[arg(long)]
        source_id: usize,
        #[arg(long)]
        target_id: usize,
        #[arg(short = 't', long)]
        relation_type: String,
        #[arg(short, long, default_value = "1.0")]
        weight: f64,
    },
    /// BFS traversal from a start node (or add doc first with --add-doc)
    GraphTraverse {
        #[arg(short, long)]
        text: String,
        #[arg(short, long)]
        embedding: Option<String>,
        #[arg(short, long, default_value = "3")]
        max_depth: usize,
        #[arg(long, default_value = "false")]
        add_doc: bool,
    },
    /// Hybrid or entity search on the graph store
    GraphSearch {
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value = "10")]
        k: usize,
        #[arg(short = 't', long, default_value = "hybrid")]
        search_type: String,
    },
    /// Print graph store statistics
    GraphStats,
    // ── ML / Entity / Data Lake Commands ──────────────────────────────
    /// Extract entities from text using structural patterns and n-grams
    ExtractEntities {
        #[arg(short, long)]
        text: String,
        #[arg(short, long, default_value = "0.3")]
        min_score: f64,
    },
    /// List datasets in the data lake
    LakeList,
    /// Register a dataset in the data lake
    LakeRegister {
        #[arg(long)]
        id: String,
        #[arg(long)]
        name: String,
        #[arg(long, default_value = "0")]
        n_vectors: usize,
        #[arg(long, default_value = "640")]
        dim: usize,
        #[arg(short, long)]
        description: Option<String>,
    },
    /// Evaluate embedding quality with synthetic benchmark
    EvalEmbeddings {
        #[arg(short, long, default_value = "64")]
        dim: usize,
        #[arg(short = 'q', long, default_value = "10")]
        n_queries: usize,
    },
    /// Benchmark HNSW search with recall measurement (single-process)
    BenchHnsw {
        /// Binary file with training vectors [u64 rows][u64 cols][f32 data]
        #[arg(short, long)]
        train: PathBuf,
        /// Binary file with query vectors [u64 rows][u64 cols][f32 data]
        #[arg(long)]
        queries: PathBuf,
        /// Binary file with ground truth [u64 n_queries][u64 k][i64 indices]
        #[arg(long)]
        gt: Option<PathBuf>,
        /// Vector dimension
        #[arg(short = 'd', long)]
        dim: usize,
        /// Number of nearest neighbors to retrieve
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// Number of queries to test (default: all)
        #[arg(short = 'n', long)]
        samples: Option<usize>,
        /// Data directory for loading/saving HNSW index
        #[arg(long)]
        data_dir: Option<String>,
        /// Max splat capacity
        #[arg(long, default_value = "100000")]
        max_splats: usize,
        /// Distance metric: "l2" or "cosine" (default: "l2" for fair SIFT comparison)
        #[arg(long, default_value = "l2")]
        metric: String,
        /// HNSW ef_search override (default: 50)
        #[arg(long)]
        ef_search: Option<usize>,
        /// HNSW ef_construction override (default: 200)
        #[arg(long)]
        ef_construction: Option<usize>,
        /// Over-fetch multiplier for re-ranking (default: 2)
        #[arg(long, default_value = "2")]
        over_fetch: usize,
    },
    /// Search with spatial memory filters (Wing/Room/Hall)
    SpatialSearch {
        /// Query text or comma-separated vector
        #[arg(short, long)]
        query: String,
        /// Filter by wing (project/persona/domain)
        #[arg(short, long)]
        wing: Option<String>,
        /// Filter by room (semantic cluster label)
        #[arg(short, long)]
        room: Option<String>,
        /// Filter by hall (memory type: fact, decision, event, error)
        #[arg(short, long)]
        hall: Option<String>,
        /// Number of results
        #[arg(short, long, default_value = "10")]
        k: usize,
    },
    /// Show spatial memory structure (wings, rooms, tunnels)
    SpatialInfo,
    // ── Cluster Commands ──────────────────────────────────────────────
    /// Join a node to the cluster
    ClusterJoin {
        /// Edge node ID
        #[arg(short, long)]
        id: String,
        /// Node URL (e.g. localhost:8001)
        #[arg(short, long)]
        url: String,
        /// Node role: worker, edge, coordinator
        #[arg(short = 't', long, default_value = "worker")]
        role: String,
        /// Node weight for routing (default: 1.0)
        #[arg(short, long, default_value = "1.0")]
        weight: f64,
    },
    /// Remove a node from the cluster
    ClusterLeave {
        /// Edge node ID to remove
        #[arg(short, long)]
        id: String,
    },
    /// Show cluster status — nodes, routing, sharding
    ClusterStatus {
        /// Show detailed per-node info
        #[arg(short, long)]
        verbose: bool,
    },
    /// Ingest documents to cluster with sharding
    ClusterIngest {
        /// Document ID prefix
        #[arg(short, long, default_value = "doc")]
        prefix: String,
        /// Number of documents to ingest
        #[arg(short = 'n', long, default_value = "1000")]
        n_docs: usize,
        /// Sharding strategy: hash, cluster, geo
        #[arg(short, long, default_value = "hash")]
        strategy: String,
    },
    /// Search the cluster with routing and RRF aggregation
    ClusterSearch {
        /// Query vector (comma-separated floats)
        #[arg(short, long)]
        query: String,
        /// Number of results
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// Routing strategy: broadcast, round_robin, least_loaded
        #[arg(short = 't', long, default_value = "broadcast")]
        strategy: String,
    },
    /// Benchmark cluster search performance
    ClusterBench {
        /// Number of queries to run
        #[arg(short = 'n', long, default_value = "1000")]
        n_queries: usize,
        /// Number of results per query
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// Routing strategy: broadcast, round_robin, least_loaded
        #[arg(short = 't', long, default_value = "broadcast")]
        strategy: String,
    },
    /// Reset cluster state (remove all nodes)
    ClusterReset,
    // ── Verbatim Storage ─────────────────────────────────────────────
    /// Store document with verbatim text for exact recall
    VerbatimStore {
        /// Document ID
        #[arg(short, long)]
        id: String,
        /// Document text content (verbatim)
        #[arg(short, long)]
        text: String,
        /// Optional category tag
        #[arg(short, long)]
        category: Option<String>,
    },
    /// Retrieve original document text by ID
    VerbatimGet {
        /// Document ID to retrieve
        #[arg(short, long)]
        id: String,
    },
    /// Search with verbatim text and confidence scores
    VerbatimSearch {
        /// Search query text
        #[arg(short, long)]
        query: String,
        /// Number of results
        #[arg(short, long, default_value = "10")]
        k: usize,
    },
    // ── AAAK Text Compression ────────────────────────────────────────
    /// Compress text using AAAK (semantic + binary)
    Compress {
        /// Input text to compress
        #[arg(short, long)]
        text: String,
        /// Show detailed compression stats
        #[arg(short, long)]
        verbose: bool,
    },
    /// Decompress AAAK binary data (hex-encoded input)
    Decompress {
        /// Hex-encoded compressed data
        #[arg(short, long)]
        data: String,
    },
    /// Benchmark AAAK compression ratio on sample text
    CompressBench {
        /// Text size category: small, medium, large
        #[arg(short, long, default_value = "medium")]
        size: String,
    },
    /// Benchmark LongMemEval with SplatsDB native pipeline (spatial filter + vector search)
    BenchLongMemEval {
        /// Binary file with session embeddings [u64 n][u64 dim][f32 data]
        #[arg(long)]
        sessions: PathBuf,
        /// Binary file with query embeddings [u64 n][u64 dim][f32 data]
        #[arg(long)]
        queries: PathBuf,
        /// JSON metadata file (sessions, queries, answer_session_ids)
        #[arg(long)]
        meta: PathBuf,
        /// Number of results to retrieve
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// HNSW ef_search override
        #[arg(long, default_value = "100")]
        ef_search: usize,
        /// Over-fetch multiplier
        #[arg(long, default_value = "3")]
        over_fetch: usize,
    },
}

pub fn dispatch(cli: Cli) {
    let config = helpers::make_config(cli.dim, cli.max_splats);

    match cli.command {
        Commands::Index { input, shard } => {
            index_cmds::cmd_index(cli.data_dir, cli.backend, config, input, shard)
        }
        Commands::Append { input, shard } => {
            index_cmds::cmd_append(cli.data_dir, cli.backend, config, input, shard)
        }
        Commands::Search { query, k, format } => {
            search_cmds::cmd_search(cli.data_dir, config, query, k, format)
        }
        Commands::SearchFile { input, k, format } => {
            search_cmds::cmd_search_file(cli.data_dir, config, input, k, format)
        }
        Commands::Status { verbose } => {
            data_cmds::cmd_status(cli.data_dir, cli.max_splats, config, verbose)
        }
        Commands::SocCheck => data_cmds::cmd_soc_check(cli.data_dir, config),
        Commands::SocAvalanche { seed } => data_cmds::cmd_soc_avalanche(cli.data_dir, config, seed),
        Commands::SocRelax { iterations } => {
            data_cmds::cmd_soc_relax(cli.data_dir, config, iterations)
        }
        Commands::Save { shard } => data_cmds::cmd_save(cli.data_dir, config, shard),
        Commands::Load { shard } => data_cmds::cmd_load(cli.data_dir, cli.backend, shard),
        Commands::List => data_cmds::cmd_list(cli.data_dir, cli.backend),
        Commands::DocAdd { id, text, metadata } => {
            data_cmds::cmd_doc_add(cli.data_dir, cli.backend, id, text, metadata)
        }
        Commands::DocGet { id } => data_cmds::cmd_doc_get(cli.data_dir, cli.backend, id),
        Commands::DocDel { id } => data_cmds::cmd_doc_del(cli.data_dir, cli.backend, id),
        Commands::Backup { output } => data_cmds::cmd_backup(cli.data_dir, cli.backend, output),
        Commands::QuantIndex {
            input,
            bits,
            algorithm,
            seed,
        } => data_cmds::cmd_quant_index(input, bits, algorithm, seed),
        Commands::QuantSearch { query, top_k } => {
            search_cmds::cmd_quant_search(cli.data_dir, cli.backend, cli.dim, query, top_k)
        }
        Commands::QuantStatus => data_cmds::cmd_quant_status(cli.dim),
        Commands::GpuInfo => data_cmds::cmd_gpu_info(),
        Commands::BenchGpu {
            n_vectors,
            dim,
            n_queries,
            top_k,
            metric,
        } => index_cmds::cmd_bench_gpu(n_vectors, dim, n_queries, top_k, metric),
        Commands::Ingest {
            input,
            n_clusters,
            seed,
        } => index_cmds::cmd_ingest(config, input, n_clusters, seed),
        Commands::IngestHierarchical {
            input,
            n_clusters,
            min_cluster_size,
            seed,
        } => index_cmds::cmd_ingest_hierarchical(config, input, n_clusters, min_cluster_size, seed),
        Commands::IngestLeader {
            input,
            target_clusters,
            threshold,
            seed,
        } => index_cmds::cmd_ingest_leader(
            cli.data_dir,
            cli.backend,
            config,
            input,
            target_clusters,
            threshold,
            seed,
        ),
        Commands::FusedSearch {
            query,
            query_file,
            k,
        } => search_cmds::cmd_fused_search(cli.data_dir, config, query, query_file, k),
        Commands::PresetInfo { preset } => data_cmds::cmd_preset_info(preset),
        Commands::HnswSearch {
            query,
            query_file,
            k,
        } => search_cmds::cmd_hnsw_search(
            cli.data_dir,
            cli.dim,
            cli.max_splats,
            query,
            query_file,
            k,
        ),
        Commands::LshSearch {
            query,
            query_file,
            k,
        } => {
            search_cmds::cmd_lsh_search(cli.data_dir, cli.dim, cli.max_splats, query, query_file, k)
        }
        Commands::BenchGpuIngest {
            n_vectors,
            dim,
            n_clusters,
            n_queries,
        } => index_cmds::cmd_bench_gpu_ingest(n_vectors, dim, n_clusters, n_queries),
        Commands::Serve { port, host } => data_cmds::cmd_serve(port, &host),
        Commands::Mcp => splatsdb::mcp_server::run_mcp_server(),
        // ── Graph Splat ──
        Commands::GraphAddDoc { text, embedding } => graph_cmds::cmd_graph_add_doc(text, embedding),
        Commands::GraphAddEntity {
            name,
            embedding,
            entity_type,
        } => graph_cmds::cmd_graph_add_entity(name, embedding, entity_type),
        Commands::GraphAddRelation {
            source_id,
            target_id,
            relation_type,
            weight,
        } => graph_cmds::cmd_graph_add_relation(source_id, target_id, relation_type, weight),
        Commands::GraphTraverse {
            text,
            embedding,
            max_depth,
            add_doc,
        } => {
            let emb_str = embedding.unwrap_or_default();
            graph_cmds::cmd_graph_traverse(text, emb_str, max_depth, add_doc)
        }
        Commands::GraphSearch {
            query,
            k,
            search_type,
        } => graph_cmds::cmd_graph_search(query, k, search_type),
        Commands::GraphStats => graph_cmds::cmd_graph_stats(),
        // ── ML / Entity / Data Lake ──
        Commands::ExtractEntities { text, min_score } => {
            ml_cmds::cmd_extract_entities(text, min_score)
        }
        Commands::LakeList => ml_cmds::cmd_lake_list(cli.data_dir),
        Commands::LakeRegister {
            id,
            name,
            n_vectors,
            dim,
            description,
        } => ml_cmds::cmd_lake_register(cli.data_dir, id, name, n_vectors, dim, description),
        Commands::EvalEmbeddings { dim, n_queries } => ml_cmds::cmd_eval_embeddings(dim, n_queries),
        Commands::BenchHnsw {
            train,
            queries,
            gt,
            dim,
            k,
            samples,
            data_dir,
            max_splats,
            metric,
            ef_search,
            ef_construction,
            over_fetch,
        } => search_cmds::cmd_bench_hnsw(
            train,
            queries,
            gt,
            dim,
            k,
            samples,
            data_dir,
            max_splats,
            metric,
            ef_search,
            ef_construction,
            over_fetch,
        ),
        // ── Spatial Memory ──
        Commands::SpatialSearch {
            query,
            wing,
            room,
            hall,
            k,
        } => crate::cli::spatial_cmds::cmd_spatial_search(cli.data_dir, query, wing, room, hall, k),
        Commands::SpatialInfo => crate::cli::spatial_cmds::cmd_spatial_info(cli.data_dir),
        // ── Cluster ──
        Commands::ClusterJoin {
            id,
            url,
            role,
            weight,
        } => crate::cli::cluster_cmds::cmd_cluster_join(&id, &url, &role, weight),
        Commands::ClusterLeave { id } => crate::cli::cluster_cmds::cmd_cluster_leave(&id),
        Commands::ClusterStatus { verbose } => {
            crate::cli::cluster_cmds::cmd_cluster_status(verbose)
        }
        Commands::ClusterIngest {
            prefix,
            n_docs,
            strategy,
        } => crate::cli::cluster_cmds::cmd_cluster_ingest(&prefix, n_docs, &strategy),
        Commands::ClusterSearch { query, k, strategy } => {
            crate::cli::cluster_cmds::cmd_cluster_search(&query, k, &strategy)
        }
        Commands::ClusterBench {
            n_queries,
            k,
            strategy,
        } => crate::cli::cluster_cmds::cmd_cluster_bench(n_queries, k, &strategy),
        Commands::ClusterReset => crate::cli::cluster_cmds::cmd_cluster_reset(),
        // ── Verbatim Storage ──
        Commands::VerbatimStore { id, text, category } => {
            crate::cli::verbatim_cmds::cmd_verbatim_store(
                cli.data_dir,
                cli.backend,
                id,
                text,
                category,
            )
        }
        Commands::VerbatimGet { id } => {
            crate::cli::verbatim_cmds::cmd_verbatim_get(cli.data_dir, cli.backend, id)
        }
        Commands::VerbatimSearch { query, k } => {
            crate::cli::verbatim_cmds::cmd_verbatim_search(cli.data_dir, config, query, k)
        }
        // ── AAAK Compression ──
        Commands::Compress { text, verbose } => {
            crate::cli::verbatim_cmds::cmd_compress(text, verbose)
        }
        Commands::Decompress { data } => crate::cli::verbatim_cmds::cmd_decompress(data),
        Commands::CompressBench { size } => crate::cli::verbatim_cmds::cmd_compress_bench(size),
        Commands::BenchLongMemEval {
            sessions,
            queries,
            meta,
            k,
            ef_search,
            over_fetch,
        } => bench_longmemeval::cmd_bench_longmemeval(
            sessions.to_string_lossy().to_string(),
            queries.to_string_lossy().to_string(),
            meta.to_string_lossy().to_string(),
            k,
            ef_search,
            over_fetch,
        ),
    }
}
