//! CLI command handlers — extracted from main.rs

mod data_cmds;
mod helpers;
mod index_cmds;
mod search_cmds;

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
    },
    /// Start MCP server (stdio transport for AI agent integration)
    Mcp,
}

pub fn dispatch(cli: Cli) {
    let config = helpers::make_config(cli.dim, cli.max_splats);

    match cli.command {
        Commands::Index { input, shard } => index_cmds::cmd_index(cli.data_dir, cli.backend, config, input, shard),
        Commands::Search { query, k, format } => search_cmds::cmd_search(cli.data_dir, config, query, k, format),
        Commands::SearchFile { input, k, format } => search_cmds::cmd_search_file(cli.data_dir, config, input, k, format),
        Commands::Status { verbose } => data_cmds::cmd_status(cli.data_dir, cli.max_splats, config, verbose),
        Commands::SocCheck => data_cmds::cmd_soc_check(cli.data_dir, config),
        Commands::SocAvalanche { seed } => data_cmds::cmd_soc_avalanche(cli.data_dir, config, seed),
        Commands::SocRelax { iterations } => data_cmds::cmd_soc_relax(cli.data_dir, config, iterations),
        Commands::Save { shard } => data_cmds::cmd_save(cli.data_dir, config, shard),
        Commands::Load { shard } => data_cmds::cmd_load(cli.data_dir, cli.backend, shard),
        Commands::List => data_cmds::cmd_list(cli.data_dir, cli.backend),
        Commands::DocAdd { id, text, metadata } => data_cmds::cmd_doc_add(cli.data_dir, cli.backend, id, text, metadata),
        Commands::DocGet { id } => data_cmds::cmd_doc_get(cli.data_dir, cli.backend, id),
        Commands::DocDel { id } => data_cmds::cmd_doc_del(cli.data_dir, cli.backend, id),
        Commands::Backup { output } => data_cmds::cmd_backup(cli.data_dir, cli.backend, output),
        Commands::QuantIndex { input, bits, algorithm, seed } => data_cmds::cmd_quant_index(input, bits, algorithm, seed),
        Commands::QuantSearch { query, top_k } => search_cmds::cmd_quant_search(cli.data_dir, cli.backend, cli.dim, query, top_k),
        Commands::QuantStatus => data_cmds::cmd_quant_status(cli.dim),
        Commands::GpuInfo => data_cmds::cmd_gpu_info(),
        Commands::BenchGpu { n_vectors, dim, n_queries, top_k, metric } => index_cmds::cmd_bench_gpu(n_vectors, dim, n_queries, top_k, metric),
        Commands::Ingest { input, n_clusters, seed } => index_cmds::cmd_ingest(config, input, n_clusters, seed),
        Commands::IngestHierarchical { input, n_clusters, min_cluster_size, seed } => index_cmds::cmd_ingest_hierarchical(config, input, n_clusters, min_cluster_size, seed),
        Commands::IngestLeader { input, target_clusters, threshold, seed } => index_cmds::cmd_ingest_leader(cli.data_dir, cli.backend, config, input, target_clusters, threshold, seed),
        Commands::FusedSearch { query, query_file, k } => search_cmds::cmd_fused_search(cli.data_dir, config, query, query_file, k),
        Commands::PresetInfo { preset } => data_cmds::cmd_preset_info(preset),
        Commands::HnswSearch { query, query_file, k } => search_cmds::cmd_hnsw_search(cli.data_dir, cli.dim, cli.max_splats, query, query_file, k),
        Commands::LshSearch { query, query_file, k } => search_cmds::cmd_lsh_search(cli.data_dir, cli.dim, cli.max_splats, query, query_file, k),
        Commands::BenchGpuIngest { n_vectors, dim, n_clusters, n_queries } => index_cmds::cmd_bench_gpu_ingest(n_vectors, dim, n_clusters, n_queries),
        Commands::Serve { port } => data_cmds::cmd_serve(port),
        Commands::Mcp => m2m_vector_search::mcp_server::run_mcp_server(),
    }
}
