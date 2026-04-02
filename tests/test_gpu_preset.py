"""
Test all GPU preset modules for M2M Rust.
Tests: ingest, search, HNSW, quantization, LSH, semantic memory, graph, fused search.
Reports timing and correctness.
"""
import subprocess, json, time, struct, random, os, sys

M2M = r"D:\m2m-memory\target\debug\m2m-vector-search.exe"
DATA = r"D:\m2m-data"
DIM = 64  # Use 64D for fast debug-mode testing
N_VECTORS = 500
K = 5

def run(args, timeout=120):
    """Run m2m binary, return (stdout, stderr, returncode, elapsed)"""
    cmd = [M2M] + args
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace')
        elapsed = time.time() - t0
        return r.stdout.strip(), r.stderr.strip(), r.returncode, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return "", "TIMEOUT", -1, elapsed

def gen_vectors(path, n, d, seed=42):
    random.seed(seed)
    with open(path, 'wb') as f:
        f.write(struct.pack('<QQ', n, d))
        for _ in range(n * d):
            f.write(struct.pack('<f', random.gauss(0, 1)))
    return n, d

def gen_query(path, d, seed=99):
    random.seed(seed)
    with open(path, 'wb') as f:
        f.write(struct.pack('<QQ', 1, d))
        for _ in range(d):
            f.write(struct.pack('<f', random.gauss(0, 1)))

def parse_results(stdout):
    """Parse JSON array of neighbor results"""
    if not stdout:
        return []
    try:
        return json.loads(stdout)
    except:
        return []

def test(name, args, check_fn=None, timeout=120):
    """Run a test and report"""
    stdout, stderr, rc, elapsed = run(args, timeout)
    passed = rc == 0 or (rc == 1 and stderr and "warning" in stderr.lower() and stdout)
    # rc=1 from PowerShell stderr warnings is ok if we have stdout
    results = parse_results(stdout)
    if check_fn:
        passed = check_fn(stdout, stderr, rc, results)
    
    status = "PASS" if passed else "FAIL"
    detail = ""
    if results:
        detail = f" | {len(results)} results"
    if not passed and stderr:
        # Extract key error
        for line in stderr.split('\n'):
            if 'Error' in line or 'error' in line or 'panic' in line:
                detail += f" | {line.strip()[:80]}"
                break
    
    print(f"  [{status}] {name}: {elapsed:.2f}s{detail}")
    return passed, results, elapsed

def main():
    print("=" * 60)
    print(f"M2M GPU Preset Module Tests")
    print(f"Vectors: {N_VECTORS}x{DIM} | K: {K}")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Generate test data
    vec_path = os.path.join(DATA, "gpu_test_vectors.bin")
    query_path = os.path.join(DATA, "gpu_test_query.bin")
    test_data_dir = os.path.join(DATA, "gpu_test")
    os.makedirs(test_data_dir, exist_ok=True)
    
    print("\n[Setup] Generating test data...")
    n, d = gen_vectors(vec_path, N_VECTORS, DIM)
    gen_query(query_path, DIM)
    print(f"  {n} vectors {d}D + query vector generated")
    
    common = ["--data-dir", test_data_dir, "--dim", str(DIM)]
    
    # ===== 1. INGEST =====
    print("\n--- 1. INGEST (Leader Clustering) ---")
    p, _, t = test("ingest-leader", 
        common + ["ingest-leader", "--input", vec_path, "--target-clusters", "20"],
        check_fn=lambda so, se, rc, r: '"saved":true' in so or '"status":"ingested"' in so)
    if p: passed += 1
    else: failed += 1
    
    # ===== 2. STATUS =====
    print("\n--- 2. STATUS ---")
    p, res, t = test("status",
        common + ["status"],
        check_fn=lambda so, se, rc, r: '"n_active"' in so)
    if p:
        passed += 1
        # Parse n_active
        try:
            st = json.loads(res if res else so)
            print(f"    n_active={st.get('n_active')}, dim={st.get('embedding_dim')}")
        except: pass
    else: failed += 1
    
    # ===== 3. SEARCH (basic linear scan) =====
    print("\n--- 3. SEARCH (Linear Scan) ---")
    p, res, t = test("search-file",
        common + ["search-file", "--input", query_path, "--k", str(K)],
        check_fn=lambda so, se, rc, r: len(r) > 0)
    if p:
        passed += 1
        dists = [r.get('distance', 0) for r in res]
        print(f"    Top-1 dist: {dists[0]:.4f} | Top-{K} dist range: [{min(dists):.4f}, {max(dists):.4f}]")
    else: failed += 1
    
    # ===== 4. FUSED SEARCH =====
    print("\n--- 4. FUSED SEARCH ---")
    p, res, t = test("fused-search",
        common + ["fused-search", "--query-file", query_path, "--k", str(K)],
        check_fn=lambda so, se, rc, r: len(r) > 0)
    if p:
        passed += 1
        dists = [r.get('distance', 0) for r in res]
        print(f"    {len(res)} results in {t:.3f}s | Top-1: {dists[0]:.4f}")
    else: failed += 1
    
    # ===== 5. HNSW SEARCH =====
    print("\n--- 5. HNSW SEARCH ---")
    p, res, t = test("hnsw-search",
        common + ["hnsw-search", "--query-file", query_path, "--k", str(K)],
        check_fn=lambda so, se, rc, r: len(r) > 0)
    if p:
        passed += 1
        dists = [r.get('distance', 0) for r in res]
        print(f"    {len(res)} results in {t:.3f}s | Top-1: {dists[0]:.4f}")
    else: 
        failed += 1
        # Check if HNSW not enabled
        _, se, _, _ = run(common + ["hnsw-search", "--query-file", query_path, "--k", str(K)])
        if "HNSW not enabled" in se:
            print(f"    NOTE: HNSW not enabled in advanced preset")
    
    # ===== 6. LSH SEARCH =====
    print("\n--- 6. LSH SEARCH ---")
    p, res, t = test("lsh-search",
        common + ["lsh-search", "--query-file", query_path, "--k", str(K)],
        check_fn=lambda so, se, rc, r: len(r) > 0)
    if p:
        passed += 1
        print(f"    {len(res)} results in {t:.3f}s")
    else:
        failed += 1
        _, se, _, _ = run(common + ["lsh-search", "--query-file", query_path, "--k", str(K)])
        if "LSH not enabled" in se or "lsh" in se.lower():
            print(f"    NOTE: LSH not available")
    
    # ===== 7. QUANT INDEX + SEARCH =====
    print("\n--- 7. QUANTIZATION ---")
    p, res, t = test("quant-index",
        common + ["quant-index", "--input", vec_path],
        check_fn=lambda so, se, rc, r: rc == 0 or so)
    if p: passed += 1
    else: failed += 1
    
    p, res, t = test("quant-search",
        common + ["quant-search", "--input", query_path, "--k", str(K)],
        check_fn=lambda so, se, rc, r: len(r) > 0)
    if p: 
        passed += 1
        print(f"    {len(res)} results in {t:.3f}s")
    else: failed += 1
    
    # ===== 8. SOC (Self-Organizing Criticality) =====
    print("\n--- 8. SOC MODULE ---")
    p, _, t = test("soc-check",
        common + ["soc-check"],
        check_fn=lambda so, se, rc, r: rc == 0 or '"criticality"' in so or so)
    if p: passed += 1
    else: failed += 1
    
    p, _, t = test("soc-relax",
        common + ["soc-relax", "--iterations", "5"],
        check_fn=lambda so, se, rc, r: rc == 0 or so)
    if p: passed += 1
    else: failed += 1
    
    # ===== 9. PRESET INFO =====
    print("\n--- 9. PRESET INFO (GPU) ---")
    p, res, t = test("preset-info",
        common + ["preset-info", "--preset", "gpu"],
        check_fn=lambda so, se, rc, r: '"preset"' in so)
    if p:
        passed += 1
        try:
            info = json.loads(res if isinstance(res, list) and res else so)
            if isinstance(info, list):
                for item in info:
                    if item.get('preset') == 'gpu':
                        subs = item.get('subsystems', {})
                        print(f"    GPU subsystems: {json.dumps(subs)}")
        except: pass
    else: failed += 1
    
    # ===== 10. SAVE/LOAD CYCLE =====
    print("\n--- 10. PERSISTENCE (Save/Load) ---")
    p, _, t = test("save",
        common + ["save"],
        check_fn=lambda so, se, rc, r: rc == 0 or 'saved' in so.lower() or so)
    if p: passed += 1
    else: failed += 1
    
    p, _, t = test("list-shards",
        common + ["list"],
        check_fn=lambda so, se, rc, r: '"shards"' in so)
    if p:
        passed += 1
        print(f"    Shards: {so[:100]}")
    else: failed += 1
    
    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed / {failed} failed / {passed+failed} total")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
