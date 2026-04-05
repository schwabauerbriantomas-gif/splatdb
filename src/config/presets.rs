//! Preset configurations for SplatDB.
//!
//! Convenience constructors for common deployment scenarios.

use super::types::*;
use super::detect_device;
use super::SplatDBConfig;

impl SplatDBConfig {
    /// Handle device auto-detection and flags.
    pub fn finalize(&mut self) {
        if self.device == "auto" {
            self.device = detect_device().to_string();
        }

        match self.device.as_str() {
            "vulkan" => self.enable_vulkan = true,
            "cuda" => self.enable_cuda = true,
            _ => {}
        }
    }

    /// Creates a 'Simple' configuration for edge computing.
    ///
    /// Stripped-down: no quantization overhead, no graph, no auto-scaling.
    /// Pure HRM2 search with RAM-only memory.
    pub fn simple(device: Option<&str>) -> Self {
        let mut config = Self::default();

        if let Some(d) = device {
            config.device = d.to_string();
        }

        config.enable_3_tier_memory = false;
        config.memory_tier = "ram-only".to_string();
        config.splat_temperature = 0.0;

        // Disable heavy features
        config.enable_quantization = false;
        config.enable_graph = false;
        config.enable_semantic_memory = false;
        config.enable_mapreduce = false;
        config.enable_auto_scaling = false;
        config.enable_quality_reflection = false;
        config.enable_hnsw = false;
        config.enable_lsh = false;

        // Reduce HRM2 overhead
        config.hrm2_n_coarse = 8;
        config.hrm2_n_fine = 32;
        config.hrm2_n_probe = 2;
        config.max_splats = 10000;
        config.n_splats_init = 1000;
        config.knn_k = 32;

        // Reduce memory footprint
        config.langevin_steps = 0;
        config.context_local = 8;
        config.context_medium = 32;
        config.context_global = 128;
        config.hidden_dim = 512;
        config.moe_experts = 2;
        config.moe_active = 1;

        config
    }

    /// Creates an 'Advanced' configuration for Agentic AI workloads.
    ///
    /// Everything enabled: TurboQuant compression, GraphSplat knowledge graph,
    /// Semantic Memory with hybrid BM25+vector, auto-scaling, quality reflection.
    pub fn advanced(device: Option<&str>) -> Self {
        let mut config = Self::default();

        if let Some(d) = device {
            config.device = d.to_string();
        }

        config.enable_3_tier_memory = true;
        config.memory_tier = "3-tier".to_string();
        config.splat_temperature = 0.1;

        // Full quantization with TurboQuant for memory efficiency
        config.enable_quantization = true;
        config.quant_algorithm = QuantAlgorithm::TurboQuant;
        config.quant_bits = 4;
        config.quant_projections = 80;
        config.quant_search_fraction = 0.2;

        // GraphSplat for knowledge graph augmentation
        config.enable_graph = true;
        config.graph_max_neighbors = 20;
        config.graph_traverse_depth = 5;
        config.graph_boost_weight = 0.1;

        // Semantic Memory with RRF fusion and temporal decay
        config.enable_semantic_memory = true;
        config.semantic_fusion = FusionMethod::Rrf;
        config.semantic_vector_weight = 0.6;
        config.semantic_bm25_weight = 0.4;
        config.semantic_decay_halflife = 3600.0; // 1 hour

        // HNSW as secondary index for exact recall
        config.enable_hnsw = true;
        config.hnsw_m = 32;
        config.hnsw_ef_construction = 400;
        config.hnsw_ef_search = 100;

        // MapReduce for bulk indexing
        config.enable_mapreduce = true;
        config.mapreduce_n_chunks = 16;

        // Auto-scaling for distributed deployments
        config.enable_auto_scaling = true;
        config.autoscale_min_nodes = 2;
        config.autoscale_max_nodes = 20;
        config.autoscale_cooldown_secs = 30.0;

        // Quality reflection with high recall target
        config.enable_quality_reflection = true;
        config.quality_recall_target = 0.99;

        // Larger capacity
        config.max_splats = 1_000_000;
        config.n_splats_init = 50000;
        config.knn_k = 128;
        config.hrm2_n_coarse = 32;
        config.hrm2_n_fine = 128;
        config.hrm2_n_probe = 8;

        // Full context windows
        config.context_local = 24;
        config.context_medium = 128;
        config.context_global = 1024;

        config
    }

    /// Creates a 'Training' configuration for embedding model research.
    ///
    /// Enables training loops, evaluation, data lake ingestion, noise augmentation,
    /// and Matryoshka representation learning. Search features are secondary.
    pub fn training(device: Option<&str>) -> Self {
        let mut config = Self::default();

        if let Some(d) = device {
            config.device = d.to_string();
        }

        // Training enabled
        config.enable_training = true;
        config.training_epochs = 50;
        config.training_eval_interval = 50;
        config.training_save_interval = 500;
        config.training_noise_augmentation = true;
        config.training_matryoshka_dims = vec![32, 64, 128, 256, 640];
        config.training_distillation = true;

        // Data Lake for training data management
        config.enable_data_lake = true;
        config.data_lake_max_entries = 10_000_000;
        config.data_lake_compress = true;

        // Entity extraction for metadata enrichment
        config.enable_entity_extraction = true;
        config.entity_ngram_sizes = vec![2, 3, 4, 5];
        config.entity_min_confidence = 0.3;

        // Search features at moderate capacity for evaluation
        config.enable_quantization = false; // No compression during training
        config.enable_graph = false;
        config.enable_semantic_memory = false;
        config.max_splats = 500_000;
        config.n_splats_init = 50_000;
        config.knn_k = 32;
        config.hrm2_n_coarse = 16;
        config.hrm2_n_fine = 64;

        // Larger training batches and sequences
        config.batch_size = 128;
        config.seq_length = 64;
        config.learning_rate = 3e-4;
        config.noise_levels = vec![0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5];

        config
    }

    /// Creates a 'Distributed' configuration for multi-node cluster deployments.
    ///
    /// Enables cluster modules (router, balancer, sharding), auto-scaling,
    /// MapReduce indexing, and inter-node communication.
    pub fn distributed(device: Option<&str>) -> Self {
        let mut config = Self::default();

        if let Some(d) = device {
            config.device = d.to_string();
        }

        // Cluster and distributed features
        config.enable_auto_scaling = true;
        config.autoscale_min_nodes = 3;
        config.autoscale_max_nodes = 50;
        config.autoscale_cooldown_secs = 20.0;

        // MapReduce for distributed indexing
        config.enable_mapreduce = true;
        config.mapreduce_n_chunks = 32;

        // Compression for inter-node transfer efficiency
        config.enable_quantization = true;
        config.quant_algorithm = QuantAlgorithm::TurboQuant;
        config.quant_bits = 4;
        config.quant_projections = 80;
        config.quant_search_fraction = 0.15;

        // Quality monitoring across nodes
        config.enable_quality_reflection = true;
        config.quality_recall_target = 0.97;

        // Large scale
        config.max_splats = 10_000_000;
        config.n_splats_init = 500_000;
        config.knn_k = 64;

        // Semantic search for cross-node queries
        config.enable_semantic_memory = true;
        config.semantic_fusion = FusionMethod::Rrf;
        config.semantic_decay_halflife = 7200.0; // 2 hours

        config
    }

    /// Creates an 'MCP' configuration optimized for AI agent memory workloads.
    ///
    /// GPU-accelerated when available, moderate capacity, all smart features on.
    /// Designed for the MCP server use case: store/search/recall with embeddings.
    pub fn mcp(device: Option<&str>) -> Self {
        let mut config = Self::default();

        config.device = device.unwrap_or("auto").to_string();
        config.finalize();

        // GPU acceleration when available
        config.enable_gpu_search = config.enable_cuda;
        config.gpu_batch_size = 1024;
        config.gpu_auto_tune = true;

        // Moderate capacity for agent memory
        config.max_splats = 100_000;
        config.n_splats_init = 10_000;
        config.knn_k = 64;

        // HRM2 for fast approximate search
        config.hrm2_n_coarse = 32;
        config.hrm2_n_fine = 128;
        config.hrm2_n_probe = 8;

        // Quantization for memory efficiency
        config.enable_quantization = true;
        config.quant_algorithm = QuantAlgorithm::TurboQuant;
        config.quant_bits = 8;

        // Knowledge graph for semantic connections
        config.enable_graph = true;
        config.graph_max_neighbors = 20;
        config.graph_traverse_depth = 5;

        // Semantic memory with hybrid search
        config.enable_semantic_memory = true;
        config.semantic_fusion = FusionMethod::Rrf;
        config.semantic_vector_weight = 0.6;
        config.semantic_bm25_weight = 0.4;

        config
    }

    /// Creates a 'GPU' configuration for CUDA-accelerated search.
    ///
    /// Enables GPU search backends, auto-tuning, and large batch operations.
    /// Falls back to CPU if no GPU is available.
    #[allow(clippy::field_reassign_with_default)]
    pub fn gpu(device: Option<&str>) -> Self {
        let mut config = Self::default();

        config.device = device.unwrap_or("cuda").to_string();
        config.enable_cuda = true;
        config.enable_vulkan = false;

        // GPU search acceleration
        config.enable_gpu_search = true;
        config.gpu_batch_size = 4096;
        config.gpu_auto_tune = true;

        // HNSW on GPU for high-throughput ANN
        config.enable_hnsw = true;
        config.hnsw_m = 48;
        config.hnsw_ef_construction = 800;
        config.hnsw_ef_search = 200;

        // Full quantization pipeline
        config.enable_quantization = true;
        config.quant_algorithm = QuantAlgorithm::TurboQuant;
        config.quant_bits = 4;
        config.quant_projections = 160;
        config.quant_search_fraction = 0.05;

        // Graph and semantic at scale
        config.enable_graph = true;
        config.graph_max_neighbors = 32;
        config.graph_traverse_depth = 5;
        config.enable_semantic_memory = true;

        // Large capacity with GPU memory
        config.max_splats = 5_000_000;
        config.n_splats_init = 100_000;
        config.knn_k = 256;
        config.hrm2_n_coarse = 64;
        config.hrm2_n_fine = 256;
        config.hrm2_n_probe = 16;

        config
    }
}
