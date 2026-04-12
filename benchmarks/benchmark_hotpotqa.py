#!/usr/bin/env python3
"""
HotpotQA Multi-hop Benchmark — Where Knowledge Graphs Shine.

HotpotQA is the anti-BEIR: instead of finding 1 doc for 1 query,
you need to find MULTIPLE docs and CONNECT them.

Example: "Were Scott Derrickson and Ed Wood of the same nationality?"
  → Need doc about Scott Derrickson (nationality: American)
  → Need doc about Ed Wood (nationality: American)
  → Compare: same → "yes"

Pure vector search: embeds the full question, finds 1 doc about "Scott Derrickson",
misses Ed Wood. FAIL.

Knowledge graph approach:
  1. Embed query → find "Scott Derrickson" doc
  2. Extract entities → "Scott Derrickson" (person), "Ed Wood" (person)
  3. Traverse graph → find "Ed Wood" via shared connections (both filmmakers)
  4. Find Ed Wood doc → extract nationality → compare

This is where SplatsDB's GraphSplat hybrid search should outperform
pure dense retrieval. If it doesn't, we learn something important.

Metrics:
  - EM (Exact Match): correct answer string
  - F1: token-level F1 between predicted and gold answer
  - Supporting facts recall: did we find BOTH required docs?
  - Joint EM: both answer correct AND supporting facts correct

Reproducible. No cherry-picking. Same 100 questions for all variants.
"""

import argparse
import json
import time
import numpy as np
import re
from collections import defaultdict

import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def simple_tokenize(text):
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return tokens


def normalize_answer(s):
    """Lowercase, remove articles, punctuation, extra whitespace."""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\b(a|an|the)\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def compute_f1(prediction, ground_truth):
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction, ground_truth):
    """Exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# ══════════════════════════════════════════════════════════════════
# Knowledge Graph (mirror of src/graph_splat.rs)
# ══════════════════════════════════════════════════════════════════

class LightweightGraph:
    """Lightweight KG for HotpotQA — connects documents via shared entities."""
    
    def __init__(self):
        self.entities = defaultdict(set)  # entity_name -> set of doc_indices
        self.doc_entities = defaultdict(set)  # doc_index -> set of entity_names
        self.doc_neighbors = defaultdict(set)  # doc_index -> set of doc_indices (1-hop)
    
    def add_document(self, doc_idx, text):
        """Extract entities and add to graph."""
        # Named entities: capitalized sequences
        entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
        # Also significant multi-word terms
        entities |= set(re.findall(r'\b[A-Z][a-z]+\s+[a-z]+(?:ing|tion|ment|ness|ity)\b', text))
        
        for ent in entities:
            self.entities[ent].add(doc_idx)
            self.doc_entities[doc_idx].add(ent)
    
    def build_edges(self):
        """Connect docs that share entities."""
        for ent, docs in self.entities.items():
            doc_list = list(docs)
            for i in range(len(doc_list)):
                for j in range(i + 1, len(doc_list)):
                    self.doc_neighbors[doc_list[i]].add(doc_list[j])
                    self.doc_neighbors[doc_list[j]].add(doc_list[i])
    
    def get_neighbors(self, doc_idx, max_hops=2):
        """Get docs reachable within max_hops (mirrors traverse())."""
        visited = {doc_idx}
        frontier = {doc_idx}
        for _ in range(max_hops):
            next_frontier = set()
            for d in frontier:
                for n in self.doc_neighbors[d]:
                    if n not in visited:
                        visited.add(n)
                        next_frontier.add(n)
            frontier = next_frontier
        return visited
    
    def get_stats(self):
        n_entities = len(self.entities)
        n_docs = len(self.doc_entities)
        n_edges = sum(len(v) for v in self.doc_neighbors.values()) // 2
        return {"entities": n_entities, "documents": n_docs, "edges": n_edges}


def extract_answer_from_docs(question, doc_texts, model, k=3):
    """
    Simple extractive answer: find the sentence most similar to the question
    and extract the answer span.
    
    For yes/no questions: check if both supporting entities are found.
    For span questions: return the most similar phrase from top docs.
    """
    # Combine top-k docs into sentences
    sentences = []
    for text in doc_texts[:k]:
        for sent in text.split('. '):
            if len(sent) > 20:
                sentences.append(sent)
    
    if not sentences:
        return ""
    
    # Check for yes/no pattern
    q_lower = question.lower()
    is_yes_no = any(q_lower.startswith(w) for w in ['were ', 'was ', 'did ', 'is ', 'are ', 'do ', 'does ', 'has ', 'have '])
    
    if is_yes_no:
        # For comparison questions, check if we found both entities
        # This is a heuristic — a proper implementation would use NER
        return "yes" if len(sentences) >= 2 else "no"
    
    # For span questions, find most similar sentence and extract answer
    if sentences:
        sent_embs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False)
        sims = np.dot(sent_embs, q_emb.T).flatten()
        best_idx = np.argmax(sims)
        best_sent = sentences[best_idx]
        
        # Extract the most likely answer: shortest phrase that overlaps with question keywords
        q_tokens = set(simple_tokenize(question))
        best_tokens = best_sent.split()
        
        # Find the longest contiguous span NOT in the question
        answer_candidates = []
        current = []
        for token in best_tokens:
            if token.lower().strip('.,;:!?') not in q_tokens:
                current.append(token)
            else:
                if current:
                    answer_candidates.append(' '.join(current))
                current = []
        if current:
            answer_candidates.append(' '.join(current))
        
        if answer_candidates:
            # Return the longest candidate (most informative)
            return max(answer_candidates, key=len)
    
    return ""


# ══════════════════════════════════════════════════════════════════
# Retrieval variants
# ══════════════════════════════════════════════════════════════════

def retrieve_dense(query_emb, index, doc_ids, k=10):
    """Pure dense retrieval."""
    scores, indices = index.search(query_emb.astype(np.float32), min(k, len(doc_ids)))
    results = []
    for j in range(min(k, len(doc_ids))):
        if indices[0][j] >= 0:
            results.append((doc_ids[indices[0][j]], float(scores[0][j])))
    return results


def retrieve_graph_augmented(query_emb, index, doc_ids, graph, doc_texts_map, k=10):
    """
    Dense + Graph expansion.
    
    1. Dense search top-3 seed docs
    2. For each seed, traverse graph (2 hops) to find connected docs
    3. Add graph-expanded docs with boosted score
    4. Rerank by combined score
    """
    # Step 1: Dense seeds
    seeds = retrieve_dense(query_emb, index, doc_ids, k=3)
    
    seed_indices = [doc_ids.index(did) for did, _ in seeds if did in doc_ids]
    
    # Step 2: Graph expansion
    expanded = set()
    for idx in seed_indices:
        neighbors = graph.get_neighbors(idx, max_hops=2)
        expanded |= neighbors
    
    # Step 3: Score expanded docs
    # Seeds get dense score, expanded docs get graph score
    results = {did: score for did, score in seeds}
    
    for idx in expanded:
        if idx < len(doc_ids):
            did = doc_ids[idx]
            if did not in results:
                # Graph score: based on number of shared entities with seeds
                shared = 0
                for seed_idx in seed_indices:
                    shared += len(graph.doc_entities[idx] & graph.doc_entities[seed_idx])
                graph_score = min(shared * 0.1, 0.5)
                results[did] = graph_score
    
    # Sort: seeds first (high dense score), then graph-expanded
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    return sorted_results[:k]


def retrieve_entity_first(question, query_emb, index, doc_ids, graph, k=10):
    """
    Entity-first retrieval — designed for multi-hop.
    
    1. Extract entities from question (e.g., "Scott Derrickson", "Ed Wood")
    2. Find docs that mention EACH entity separately
    3. Merge results ensuring BOTH entities are represented
    4. This is what a human would do: "Let me look up each person"
    """
    # Extract entities from question
    entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question))
    # Filter out common non-entities
    stopwords = {'The', 'A', 'An', 'Is', 'Was', 'Were', 'Did', 'Does', 'Do',
                 'In', 'On', 'At', 'To', 'For', 'Of', 'And', 'But', 'Or'}
    entities -= stopwords
    
    # For each entity, find docs via graph
    entity_docs = {}  # entity -> set of doc_ids
    for ent in entities:
        if ent in graph.entities:
            entity_docs[ent] = {doc_ids[i] for i in graph.entities[ent] if i < len(doc_ids)}
    
    # Dense search as baseline
    dense_results = retrieve_dense(query_emb, index, doc_ids, k=k * 2)
    dense_dict = dict(dense_results)
    
    # Merge: prioritize docs that cover MORE entities
    results = {}
    for did, score in dense_results:
        results[did] = score
    
    # Add entity-specific docs with boost
    for ent, docs in entity_docs.items():
        for did in docs:
            if did in results:
                results[did] += 0.15  # Boost for matching entity
            else:
                results[did] = 0.15
    
    # Additional scoring: docs covering multiple entities get extra boost
    for did in results:
        entities_covered = sum(1 for ent, docs in entity_docs.items() if did in docs)
        if entities_covered >= 2:
            results[did] += 0.3  # Multi-entity bonus
    
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    return sorted_results[:k]


# ══════════════════════════════════════════════════════════════════
# Main benchmark runner
# ══════════════════════════════════════════════════════════════════

def run_hotpotqa(n_questions=100, variant="all"):
    print("=" * 70)
    print("HotpotQA Multi-hop Benchmark")
    print("Where Knowledge Graphs SHOULD outperform pure dense retrieval")
    print("=" * 70)
    
    # Load data
    print("\nLoading HotpotQA...")
    ds = load_dataset('hotpot_qa', 'fullwiki', split=f'validation[:{n_questions}]')
    
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    # Build corpus from all context docs
    print("Building corpus...")
    all_docs = {}  # title -> text
    doc_titles = []
    doc_texts = []
    
    for sample in ds:
        for i, title in enumerate(sample['context']['title']):
            if title not in all_docs:
                text = ' '.join(sample['context']['sentences'][i])
                all_docs[title] = text
    
    doc_titles = list(all_docs.keys())
    doc_texts = [all_docs[t] for t in doc_titles]
    
    print(f"Corpus: {len(doc_texts)} documents")
    
    # Build index
    print("Encoding corpus...")
    corpus_emb = model.encode(doc_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    dim = corpus_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_emb.astype(np.float32))
    
    # Build knowledge graph
    print("Building knowledge graph...")
    graph = LightweightGraph()
    t0 = time.perf_counter()
    for i, text in enumerate(doc_texts):
        graph.add_document(i, text)
    graph.build_edges()
    graph_time = time.perf_counter() - t0
    stats = graph.get_stats()
    print(f"Graph: {stats['entities']} entities, {stats['edges']} edges ({graph_time:.1f}s)")
    
    # Define variants
    variants = {
        "dense": ("Pure dense (BGE-small)", retrieve_dense),
        "graph": ("Dense + Graph traversal (2-hop)", retrieve_graph_augmented),
        "entity": ("Entity-first multi-hop retrieval", retrieve_entity_first),
    }
    
    if variant != "all":
        variants = {variant: variants[variant]}
    
    all_results = {}
    
    for vname, (vdesc, vfunc) in variants.items():
        print(f"\n[{vname}] {vdesc}")
        
        metrics = {"em": [], "f1": [], "sp_recall": [], "sp_precision": [], "sp_f1": [], "joint_em": []}
        t0 = time.perf_counter()
        
        for sample in ds:
            question = sample['question']
            answer = sample['answer']
            gold_titles = set(sample['supporting_facts']['title'])
            
            # Retrieve
            q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False)
            
            if vname == "dense":
                results = vfunc(q_emb, index, doc_titles, k=10)
            elif vname == "graph":
                results = vfunc(q_emb, index, doc_titles, graph, None, k=10)
            elif vname == "entity":
                results = vfunc(question, q_emb, index, doc_titles, graph, k=10)
            
            retrieved_titles = set(title for title, score in results)
            retrieved_texts = [all_docs[title] for title, score in results]
            
            # Answer extraction
            predicted_answer = extract_answer_from_docs(question, retrieved_texts, model)
            
            # Compute metrics
            em = compute_em(predicted_answer, answer)
            f1 = compute_f1(predicted_answer, answer)
            
            # Supporting facts metrics
            sp_overlap = retrieved_titles & gold_titles
            sp_recall = len(sp_overlap) / len(gold_titles) if gold_titles else 0
            sp_precision = len(sp_overlap) / len(retrieved_titles) if retrieved_titles else 0
            sp_f1 = 2 * sp_precision * sp_recall / (sp_precision + sp_recall) if (sp_precision + sp_recall) > 0 else 0
            
            joint_em = em * (1 if sp_recall >= 1.0 else 0)
            
            metrics["em"].append(em)
            metrics["f1"].append(f1)
            metrics["sp_recall"].append(sp_recall)
            metrics["sp_precision"].append(sp_precision)
            metrics["sp_f1"].append(sp_f1)
            metrics["joint_em"].append(joint_em)
        
        elapsed = time.perf_counter() - t0
        
        avg = {k: round(np.mean(v), 4) for k, v in metrics.items()}
        avg["time_s"] = round(elapsed, 2)
        avg["qps"] = round(n_questions / elapsed, 1)
        avg["n_questions"] = n_questions
        
        all_results[vname] = avg
        
        print(f"  Answer EM: {avg['em']:.4f}  F1: {avg['f1']:.4f}")
        print(f"  Supporting Facts Recall: {avg['sp_recall']:.4f}  F1: {avg['sp_f1']:.4f}")
        print(f"  Joint EM: {avg['joint_em']:.4f}")
        print(f"  Time: {elapsed:.2f}s ({n_questions/elapsed:.1f} QPS)")
    
    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON — Supporting Facts Recall (did we find BOTH docs?)")
    print(f"{'='*70}")
    print(f"{'Variant':<20} {'SP Recall':>12} {'SP F1':>12} {'Ans EM':>12} {'Joint EM':>12}")
    print("-" * 70)
    for vname, m in all_results.items():
        print(f"{vname:<20} {m['sp_recall']:>12.4f} {m['sp_f1']:>12.4f} {m['em']:>12.4f} {m['joint_em']:>12.4f}")
    
    # Save
    output = {
        "benchmark": "HotpotQA Multi-hop",
        "description": "Tests whether knowledge graph helps find MULTIPLE supporting documents for multi-hop questions",
        "n_questions": n_questions,
        "corpus_size": len(doc_texts),
        "variants": all_results,
        "graph_stats": stats,
    }
    
    with open("bench-data/hotpotqa_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to bench-data/hotpotqa_results.json")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HotpotQA Multi-hop Benchmark")
    parser.add_argument("--n", type=int, default=100, help="Number of questions")
    parser.add_argument("--variant", choices=["dense", "graph", "entity", "all"], default="all")
    args = parser.parse_args()
    
    run_hotpotqa(n_questions=args.n, variant=args.variant)
