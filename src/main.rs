//! SplatDB Vector Search CLI — AI-friendly interface.

mod cli;

use clap::Parser;

use cli::Commands;

#[derive(Parser)]
#[command(
    name = "splatdb",
    version,
    about = "SplatDB Vector Search — Gaussian Splat semantic memory"
)]
struct Cli {
    /// Storage directory (default: ./splatdb_data)
    #[arg(long, global = true, default_value = "./splatdb_data")]
    data_dir: String,

    /// Latent dimension (default: 64)
    #[arg(long, global = true, default_value = "64")]
    dim: usize,

    /// Max splats capacity
    #[arg(long, global = true, default_value = "100000")]
    max_splats: usize,

    /// Metadata storage backend: sqlite, json
    #[arg(long, global = true, default_value = "sqlite")]
    backend: String,

    #[command(subcommand)]
    command: Commands,
}

fn main() {
    let cli = Cli::parse();
    cli::dispatch(cli);
}
