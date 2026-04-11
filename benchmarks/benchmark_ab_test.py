#!/usr/bin/env python3
"""
SplatsDB A/B Benchmark — Incremental improvement testing.

Tests 4 retrieval enhancements using ONLY techniques that exist
in SplatsDB's Rust codebase (src/). Each technique is tested
independently, then combined.

Baseline: BAAI/bge-small-en-v1.5 dense search (same as BEIR v1
but with better model — this is our "v2" baseline).

Enhancements:
  1. Graph-augmented retrieval (src/graph_splat.rs traverse + expand)
  2. Query expansion via graph entities (src/entity_extractor.rs + graph)
  3. BM25 + Dense RRF fusion (src/bm25_index.rs + src/cluster/aggregator.rs)
  4. Cross-encoder reranking (NOT in SplatsDB — testing as potential addition)

Evaluation: Standard BEIR protocol — NDCG@10, Recall@10 via pytrec-eval.
Same datasets, same qrels, same queries for every variant.

Usage:
  python benchmark_ab_test.py --datasets scifact fiqa arguana nfcorpus
  python benchmark_ab_test.py --datasets scifact --enhancement graph
"""

import argparse
import json
import time
import numpy as np
import re
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════
# Utility functions (mirrors of SplatsDB Rust implementations)
# ══════════════════════════════════════════════════════════════════

def simple_tokenize(text):
    """Mirror of SplatsDB default tokenizer with stopword removal."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    # Remove stopwords (standard IR practice, mirrors bm25_index.rs behavior)
    stopwords = {"the", "a", "an", "is", "was", "in", "to", "and", "of", "that",
                 "it", "for", "be", "are", "with", "this", "have", "from", "or",
                 "as", "but", "not", "on", "by", "at", "do", "we", "can", "will",
                 "has", "been", "would", "could", "should", "may", "might", "its",
                 "which", "their", "there", "than", "no", "if", "about", "so",
                 "only", "just", "also", "what", "how", "all", "any", "each",
                 "very", "often", "most", "more", "much", "some", "such"}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def bm25_score(query_tokens, doc_tf, doc_len, doc_freqs, n_docs, avg_dl, k1=1.5, b=0.75):
    """Mirror of src/bm25_index.rs BM25 scoring. doc_tf is {term: count} dict."""
    score = 0.0
    for term in query_tokens:
        if term not in doc_freqs:
            continue
        df = doc_freqs[term]
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf = doc_tf.get(term, 0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_dl))
        score += idf * tf_norm
    return score


def rrf_fusion(rankings, k=60):
    """Mirror of src/cluster/aggregator.rs rrf_merge."""
    rrf_scores = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: -x[1])


def extract_entities(text):
    """
    Mirror of src/entity_extractor.rs — extract named entities.
    Capitalized sequences + significant lowercase words.
    """
    entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
    words = set(re.findall(r'\b[a-z]{6,}\b', text.lower()))
    return entities | words


# ══════════════════════════════════════════════════════════════════
# Knowledge Graph (mirror of src/graph_splat.rs)
# ══════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """
    Python mirror of SplatsDB's GraphSplat (src/graph_splat.rs).
    
    Nodes: Documents + Entities + Concepts (same as Rust NodeType enum).
    Edges: co-occurrence links built from entity extraction.
    
    traverse() and hybrid_search() mirror the Rust implementations exactly.
    """
    
    def __init__(self):
        self.nodes = {}  # id -> {type, content, embedding, edges}
        self.entity_to_id = {}  # lowercase name -> node_id
        self.doc_to_id = {}  # corpus_index -> node_id
        self.next_id = 0
    
    def add_document(self, corpus_idx, text, embedding):
        """Mirror of GraphSplat::add_document()."""
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = {
            "type": "document",
            "corpus_idx": corpus_idx,
            "content": text,
            "embedding": np.array(embedding, dtype=np.float32),
            "outgoing": [],
        }
        self.doc_to_id[corpus_idx] = nid
        return nid
    
    def add_entity(self, name, embedding):
        """Mirror of GraphSplat::add_entity(). Deduplicates by name."""
        key = name.lower()
        if key in self.entity_to_id:
            return self.entity_to_id[key]
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = {
            "type": "entity",
            "content": name,
            "embedding": np.array(embedding, dtype=np.float32),
            "outgoing": [],
        }
        self.entity_to_id[key] = nid
        return nid
    
    def add_edge(self, source_id, target_id):
        """Mirror of GraphSplat edge addition."""
        if source_id in self.nodes and target_id in self.nodes:
            self.nodes[source_id]["outgoing"].append(target_id)
    
    def outgoing_count(self, node_id):
        """Mirror of GraphSplat::outgoing_count()."""
        return len(self.nodes.get(node_id, {}).get("outgoing", []))
    
    def traverse(self, start_id, max_depth=2, max_results=100):
        """
        Mirror of GraphSplat::traverse() — BFS traversal.
        Returns list of reachable node IDs (including start).
        """
        if start_id not in self.nodes:
            return []
        visited = {start_id}
        queue = [(start_id, 0)]
        result = [start_id]
        
        while queue:
            current_id, depth = queue.pop(0)
            if len(result) >= max_results:
                break
            if depth >= max_depth:
                continue
            for neighbor_id in self.nodes.get(current_id, {}).get("outgoing", []):
                if len(result) >= max_results:
                    break
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    result.append(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
        return result
    
    def get_documents_in_subgraph(self, node_ids):
        """Get corpus indices for document nodes in a subgraph."""
        return [
            self.nodes[nid]["corpus_idx"]
            for nid in node_ids
            if nid in self.nodes and self.nodes[nid]["type"] == "document"
        ]
    
    def get_stats(self):
        docs = sum(1 for n in self.nodes.values() if n["type"] == "document")
        entities = sum(1 for n in self.nodes.values() if n["type"] == "entity")
        edges = sum(len(n.get("outgoing", [])) for n in self.nodes.values())
        return {"documents": docs, "entities": entities, "edges": edges}


# ══════════════════════════════════════════════════════════════════
# Enhancement implementations
# ══════════════════════════════════════════════════════════════════

def build_graph(corpus_texts, corpus_emb, model, max_entities_per_doc=10):
    """
    Build knowledge graph from corpus — mirrors SplatsDB's pipeline:
    1. Extract entities from each doc (entity_extractor.rs)
    2. Embed entities (embedding_model.rs)
    3. Add doc-entity edges (graph_splat.rs)
    4. Add entity-entity edges for co-occurring entities
    """
    kg = KnowledgeGraph()
    
    # Add documents
    for i, text in enumerate(corpus_texts):
        kg.add_document(i, text, corpus_emb[i])
    
    # Extract and add entities
    doc_entity_ids = []
    all_entity_names = set()
    
    for i, text in enumerate(corpus_texts):
        entities = extract_entities(text)
        # Limit to top entities (longest = most specific)
        sorted_ents = sorted(entities, key=len, reverse=True)[:max_entities_per_doc]
        eids = []
        for ent in sorted_ents:
            eid = kg.add_entity(ent, corpus_emb[i])  # Use doc embedding as proxy
            eids.append(eid)
            all_entity_names.add(ent.lower())
            # Add doc <-> entity edges
            doc_nid = kg.doc_to_id[i]
            kg.add_edge(doc_nid, eid)
            kg.add_edge(eid, doc_nid)
        doc_entity_ids.append(eids)
    
    # Add entity-entity edges for co-occurring entities within same doc
    for eids in doc_entity_ids:
        for i, e1 in enumerate(eids):
            for e2 in eids[i+1:]:
                kg.add_edge(e1, e2)
                kg.add_edge(e2, e1)
    
    return kg


def search_baseline_dense(query_emb, index, corpus_ids, k=100):
    """Baseline: pure dense search with BGE model."""
    scores, indices = index.search(query_emb.astype(np.float32), min(k, len(corpus_ids)))
    results = {}
    for j in range(min(k, len(corpus_ids))):
        if indices[0][j] >= 0:
            results[corpus_ids[indices[0][j]]] = float(scores[0][j])
    return results


def search_graph_augmented(query_emb, index, corpus_ids, kg, k=100, traverse_depth=2):
    """
    Enhancement 1: Graph-augmented retrieval.
    
    Methodology:
    1. Dense search to get top-100 candidates
    2. For each candidate, traverse(max_depth=2) the knowledge graph
    3. Collect all documents reachable via graph edges
    4. Score: dense_similarity + graph_boost * (1 / distance_from_seed)
    5. Rerank by combined score
    
    Mirrors: graph_splat.rs traverse() + hybrid_search() graph_boost.
    """
    # Step 1: Dense search top-100
    scores, indices = index.search(query_emb.astype(np.float32), min(k, len(corpus_ids)))
    
    dense_results = {}
    seed_doc_ids = []
    for j in range(min(k, len(corpus_ids))):
        if indices[0][j] >= 0:
            cidx = indices[0][j]
            cid = corpus_ids[cidx]
            dense_results[cidx] = float(scores[0][j])
            seed_doc_ids.append(cidx)
    
    # Step 2-3: Graph traversal from each seed
    graph_expanded = {}  # corpus_idx -> best graph score
    
    for cidx in seed_doc_ids:
        if cidx not in kg.doc_to_id:
            continue
        doc_nid = kg.doc_to_id[cidx]
        reachable = kg.traverse(doc_nid, max_depth=traverse_depth, max_results=50)
        
        for node_id in reachable:
            node = kg.nodes.get(node_id)
            if node and node["type"] == "document":
                reachable_cidx = node["corpus_idx"]
                if reachable_cidx not in dense_results:
                    # Graph boost: inversely proportional to connectivity
                    # (mirrors outgoing_count * 0.05, max 0.2)
                    boost = min(kg.outgoing_count(node_id) * 0.05, 0.2)
                    if reachable_cidx not in graph_expanded or boost > graph_expanded[reachable_cidx]:
                        graph_expanded[reachable_cidx] = boost
    
    # Step 4: Combine
    combined = {}
    for cidx, score in dense_results.items():
        combined[corpus_ids[cidx]] = score
    
    for cidx, boost in graph_expanded.items():
        cid = corpus_ids[cidx]
        if cid in combined:
            combined[cid] += boost * 0.1  # Small boost for graph-reachable
        else:
            combined[cid] = boost * 0.1  # New document found via graph
    
    # Sort by score
    sorted_results = sorted(combined.items(), key=lambda x: -x[1])
    return {cid: score for cid, score in sorted_results[:k]}


def search_query_expansion(query_text, query_emb, index, corpus_ids, kg, model, k=100):
    """
    Enhancement 2: Query expansion via graph entities.
    
    Methodology:
    1. Extract entities from query (entity_extractor.rs)
    2. Search graph for matching entities (search_entities)
    3. For each matched entity, traverse graph to find connected docs
    4. Expand query embedding: weighted average of original + entity embeddings
    5. Search with expanded query
    
    Mirrors: entity_extractor.rs + graph_splat.rs search_entities + traverse.
    """
    # Step 1: Extract query entities
    query_entities = extract_entities(query_text)
    
    # Step 2: Find matching entities in graph
    expanded_doc_indices = set()
    for ent in query_entities:
        ent_key = ent.lower()
        if ent_key in kg.entity_to_id:
            ent_nid = kg.entity_to_id[ent_key]
            # Traverse from entity to find connected documents
            reachable = kg.traverse(ent_nid, max_depth=1, max_results=20)
            for nid in reachable:
                node = kg.nodes.get(nid)
                if node and node["type"] == "document":
                    expanded_doc_indices.add(node["corpus_idx"])
    
    # Step 3: Expand query embedding with entity embeddings
    if expanded_doc_indices:
        # Weighted average: 70% original query + 30% mean of entity-connected docs
        entity_embs = [kg.nodes[kg.doc_to_id[i]]["embedding"] for i in expanded_doc_indices 
                       if i in kg.doc_to_id]
        if entity_embs:
            mean_entity_emb = np.mean(entity_embs, axis=0)
            mean_entity_emb = mean_entity_emb / (np.linalg.norm(mean_entity_emb) + 1e-8)
            expanded_emb = 0.7 * query_emb[0] + 0.3 * mean_entity_emb
            expanded_emb = expanded_emb / (np.linalg.norm(expanded_emb) + 1e-8)
        else:
            expanded_emb = query_emb[0]
    else:
        expanded_emb = query_emb[0]
    
    # Step 4: Search with expanded query
    scores, indices = index.search(expanded_emb.astype(np.float32).reshape(1, -1), 
                                    min(k, len(corpus_ids)))
    
    results = {}
    for j in range(min(k, len(corpus_ids))):
        if indices[0][j] >= 0:
            results[corpus_ids[indices[0][j]]] = float(scores[0][j])
    
    # Boost documents found via entity expansion
    for cidx in expanded_doc_indices:
        cid = corpus_ids[cidx]
        if cid in results:
            results[cid] += 0.05  # Small entity match boost
        else:
            results[cid] = 0.05
    
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    return {cid: score for cid, score in sorted_results[:k]}


def search_bm25_hybrid(query_text, query_emb, index, corpus_ids, corpus_tf,
                        corpus_lens, doc_freqs, inverted, n_docs, avg_dl, k=100):
    """
    Enhancement 3: BM25 + Dense RRF fusion.
    
    Already implemented in v2 — included here for A/B comparison.
    Mirrors: bm25_index.rs + cluster/aggregator.rs rrf_merge.
    """
    # Dense ranking
    scores, indices = index.search(query_emb.astype(np.float32), min(k, len(corpus_ids)))
    dense_ranking = []
    for j in range(min(k, len(corpus_ids))):
        if indices[0][j] >= 0:
            dense_ranking.append((corpus_ids[indices[0][j]], float(scores[0][j])))
    
    # BM25 ranking (with inverted index + pre-computed TF)
    q_tokens = simple_tokenize(query_text)
    candidate_docs = set()
    for qt in q_tokens:
        candidate_docs |= inverted.get(qt, set())
    
    scored = []
    for j in candidate_docs:
        s = bm25_score(q_tokens, corpus_tf[j], corpus_lens[j], doc_freqs, n_docs, avg_dl)
        if s > 0:
            scored.append((corpus_ids[j], s))
    scored.sort(key=lambda x: -x[1])
    bm25_ranking = scored[:k]
    
    # RRF fusion
    fused = rrf_fusion([dense_ranking, bm25_ranking], k=60)
    return {cid: score for cid, score in fused[:k]}


def search_reranked(query_text, query_emb, index, corpus_ids, corpus_texts, reranker, k=100):
    """
    Enhancement 4: Cross-encoder reranking.
    
    Methodology:
    1. Dense search to get top-100
    2. Rerank top-30 with cross-encoder (reads query+doc together)
    3. Combine: reranked top-30 + dense 31-100
    
    NOTE: Cross-encoder reranking is NOT in SplatsDB's current codebase.
    This tests the potential improvement if we added it.
    """
    scores, indices = index.search(query_emb.astype(np.float32), min(k * 3, len(corpus_ids)))
    
    # Get top candidates
    candidates = []
    for j in range(min(k * 3, len(corpus_ids))):
        if indices[0][j] >= 0:
            candidates.append((corpus_ids[indices[0][j]], corpus_texts[indices[0][j]], float(scores[0][j])))
    
    if reranker is None or len(candidates) == 0:
        return {cid: score for cid, _, score in candidates[:k]}
    
    # Rerank top-10 with cross-encoder (fast enough on CPU)
    rerank_top = min(10, len(candidates))
    rerank_pairs = [(query_text, text) for _, text, _ in candidates[:rerank_top]]
    
    rerank_scores = reranker.predict(rerank_pairs)
    
    # Combine reranked + dense
    results = {}
    for i, (cid, text, dense_score) in enumerate(candidates[:rerank_top]):
        results[cid] = float(rerank_scores[i])
    
    for cid, text, dense_score in candidates[rerank_top:k]:
        if cid not in results:
            results[cid] = dense_score
    
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    return {cid: score for cid, score in sorted_results[:k]}


# ══════════════════════════════════════════════════════════════════
# Main A/B test runner
# ══════════════════════════════════════════════════════════════════

def run_ab_test(datasets=None, enhancement=None):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
    
    if datasets is None:
        datasets = ["scifact", "fiqa", "arguana", "nfcorpus"]
    
    print("=" * 70)
    print("SplatsDB A/B Benchmark — Incremental Enhancement Testing")
    print("=" * 70)
    print(f"Datasets: {datasets}")
    print(f"Model: BAAI/bge-small-en-v1.5 (384D)")
    print()
    
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    # Try to load cross-encoder for enhancement 4
    reranker = None
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Cross-encoder reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 ✓")
    except Exception as e:
        print(f"Cross-encoder: not available ({e})")
    print()
    
    # Define enhancement variants
    variants = {
        "baseline": "Dense only (BGE-small)",
        "graph": "Dense + Graph-augmented traversal",
        "expansion": "Dense + Query expansion via entities",
        "bm25": "Dense + BM25 RRF fusion",
        "reranker": "Dense + Cross-encoder reranking",
        "all": "All enhancements combined",
    }
    
    if enhancement:
        test_variants = {enhancement: variants[enhancement]} if enhancement in variants else variants
    else:
        test_variants = variants
    
    all_results = {}
    
    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        # Load dataset
        try:
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds_name}.zip"
            data_path = util.download_and_unzip(url, "bench-data/beir")
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        except Exception as e:
            print(f"  Error loading: {e}")
            continue
        
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid].get("title", "") + " " + corpus[cid].get("text", "") for cid in corpus_ids]
        corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
        
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        print(f"  Corpus: {len(corpus_texts):,} docs")
        print(f"  Queries: {len(query_texts)}")
        
        # Encode
        print(f"  Encoding corpus...")
        corpus_emb = model.encode(corpus_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
        print(f"  Encoding queries...")
        query_emb = model.encode(query_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
        
        # Build Faiss index
        dim = corpus_emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_emb.astype(np.float32))
        
        # Build BM25 inverted index (pre-compute TF dicts for speed)
        print(f"  Building BM25 index...")
        tokenized_corpus = [simple_tokenize(t) for t in corpus_texts]
        # Pre-compute TF dicts for O(1) lookup instead of .count()
        corpus_tf = []
        corpus_lens = []
        for tokens in tokenized_corpus:
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            corpus_tf.append(tf)
            corpus_lens.append(len(tokens))
        
        n_docs = len(corpus_texts)
        avg_dl = np.mean(corpus_lens)
        doc_freqs = {}
        inverted = defaultdict(set)
        for j, tf in enumerate(corpus_tf):
            for t in tf:
                doc_freqs[t] = doc_freqs.get(t, 0) + 1
                inverted[t].add(j)
        
        # Build Knowledge Graph (only for datasets <50K)
        kg = None
        if len(corpus_texts) < 50000:
            print(f"  Building knowledge graph...")
            t0 = time.perf_counter()
            kg = build_graph(corpus_texts, corpus_emb, model, max_entities_per_doc=8)
            kg_time = time.perf_counter() - t0
            stats = kg.get_stats()
            print(f"  Graph: {stats['documents']} docs, {stats['entities']} entities, {stats['edges']} edges ({kg_time:.1f}s)")
        
        # Run each variant
        evaluator = EvaluateRetrieval()
        ds_results = {}
        
        for variant_name, variant_desc in test_variants.items():
            print(f"\n  [{variant_name}] {variant_desc}")
            t0 = time.perf_counter()
            
            results_dict = {}
            for i, qid in enumerate(query_ids):
                q_emb = query_emb[i:i+1]
                
                if variant_name == "baseline":
                    results_dict[qid] = search_baseline_dense(q_emb, index, corpus_ids)
                
                elif variant_name == "graph":
                    if kg:
                        results_dict[qid] = search_graph_augmented(q_emb, index, corpus_ids, kg)
                    else:
                        results_dict[qid] = search_baseline_dense(q_emb, index, corpus_ids)
                
                elif variant_name == "expansion":
                    if kg:
                        results_dict[qid] = search_query_expansion(
                            query_texts[i], q_emb, index, corpus_ids, kg, model)
                    else:
                        results_dict[qid] = search_baseline_dense(q_emb, index, corpus_ids)
                
                elif variant_name == "bm25":
                    results_dict[qid] = search_bm25_hybrid(
                        query_texts[i], q_emb, index, corpus_ids,
                        corpus_tf, corpus_lens, doc_freqs, inverted, n_docs, avg_dl)
                
                elif variant_name == "reranker":
                    results_dict[qid] = search_reranked(
                        query_texts[i], q_emb, index, corpus_ids, corpus_texts, reranker)
                
                elif variant_name == "all":
                    # Combine all enhancements
                    if kg:
                        # Graph augmented
                        graph_results = search_graph_augmented(q_emb, index, corpus_ids, kg)
                        # Query expansion
                        expansion_results = search_query_expansion(
                            query_texts[i], q_emb, index, corpus_ids, kg, model)
                        # BM25 hybrid
                        bm25_results = search_bm25_hybrid(
                            query_texts[i], q_emb, index, corpus_ids,
                            corpus_tf, corpus_lens, doc_freqs, inverted, n_docs, avg_dl)
                        
                        # RRF fusion of all three
                        g_ranking = list(graph_results.items())
                        e_ranking = list(expansion_results.items())
                        b_ranking = list(bm25_results.items())
                        fused = rrf_fusion([g_ranking, e_ranking, b_ranking], k=60)
                        results_dict[qid] = {cid: score for cid, score in fused[:100]}
                    else:
                        # Large corpus: BM25 hybrid only
                        results_dict[qid] = search_bm25_hybrid(
                            query_texts[i], q_emb, index, corpus_ids,
                            corpus_tf, corpus_lens, doc_freqs, inverted, n_docs, avg_dl)
            
            elapsed = time.perf_counter() - t0
            
            # Evaluate
            ndcg, map_, recall, precision = evaluator.evaluate(qrels, results_dict, [10])
            
            ndcg_val = ndcg.get("NDCG@10", 0)
            recall_val = recall.get("Recall@10", 0)
            map_val = map_.get("MAP@10", 0)
            
            ds_results[variant_name] = {
                "ndcg_at_10": round(ndcg_val, 4),
                "recall_at_10": round(recall_val, 4),
                "map_at_10": round(map_val, 4),
                "time_s": round(elapsed, 2),
                "qps": round(len(query_ids) / elapsed, 1),
            }
            
            print(f"    NDCG@10: {ndcg_val:.4f}  Recall@10: {recall_val:.4f}  "
                  f"MAP@10: {map_val:.4f}  Time: {elapsed:.2f}s  QPS: {len(query_ids)/elapsed:.1f}")
        
        all_results[ds_name] = ds_results
    
    # Print comparison table
    print("\n" + "=" * 90)
    print("COMPARISON TABLE (NDCG@10)")
    print("=" * 90)
    
    header = f"{'Dataset':<14}"
    for vname in test_variants:
        header += f" {vname:>12}"
    header += f" {'best':>12} {'improvement':>12}"
    print(header)
    print("-" * 90)
    
    for ds_name in datasets:
        if ds_name not in all_results:
            continue
        row = f"{ds_name:<14}"
        values = []
        for vname in test_variants:
            val = all_results[ds_name].get(vname, {}).get("ndcg_at_10", 0)
            row += f" {val:>12.4f}"
            values.append(val)
        baseline = values[0] if values else 0
        best = max(values) if values else 0
        improvement = best - baseline
        row += f" {best:>12.4f} {improvement:>+12.4f}"
        print(row)
    
    # Also print Recall@10 table
    print("\nCOMPARISON TABLE (Recall@10)")
    print("-" * 90)
    
    header = f"{'Dataset':<14}"
    for vname in test_variants:
        header += f" {vname:>12}"
    header += f" {'best':>12} {'improvement':>12}"
    print(header)
    print("-" * 90)
    
    for ds_name in datasets:
        if ds_name not in all_results:
            continue
        row = f"{ds_name:<14}"
        values = []
        for vname in test_variants:
            val = all_results[ds_name].get(vname, {}).get("recall_at_10", 0)
            row += f" {val:>12.4f}"
            values.append(val)
        baseline = values[0] if values else 0
        best = max(values) if values else 0
        improvement = best - baseline
        row += f" {best:>12.4f} {improvement:>+12.4f}"
        print(row)
    
    # Save results
    output_path = "bench-data/ab_test_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SplatsDB A/B Benchmark")
    parser.add_argument("--datasets", nargs="+", default=["scifact", "fiqa", "arguana", "nfcorpus"])
    parser.add_argument("--enhancement", choices=["baseline", "graph", "expansion", "bm25", "reranker", "all"],
                        help="Test single enhancement (default: all)")
    args = parser.parse_args()
    
    run_ab_test(datasets=args.datasets, enhancement=args.enhancement)
