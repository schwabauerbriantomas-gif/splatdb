"""
Test all M2M Rust presets.
Strategy: preset-info for subsystem mapping, then test each enabled feature per preset.
"""
import subprocess, json, time, struct, random, os, sys

M2M = r"D:\m2m-memory\target\debug\m2m-vector-search.exe"
BASE_DATA = r"D:\m2m-data"
DIM = 64
N_VECTORS = 500
K = 5
PRESETS = ["simple", "advanced", "gpu", "training"]

def run(args, timeout=120):
    t0 = time.time()
    try:
        r = subprocess.run([M2M] + args, capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace')
        return r.stdout.strip(), r.stderr.strip(), r.returncode, time.time() - t0
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1, time.time() - t0

def gen_vectors(path, n, d, seed=42):
    random.seed(seed)
    with open(path, 'wb') as f:
        f.write(struct.pack('<QQ', n, d))
        for _ in range(n * d):
            f.write(struct.pack('<f', random.gauss(0, 1)))

def gen_query(path, d, seed=99):
    random.seed(seed)
    with open(path, 'wb') as f:
        f.write(struct.pack('<QQ', 1, d))
        for _ in range(d):
            f.write(struct.pack('<f', random.gauss(0, 1)))

def parse_json(stdout):
    if not stdout: return None
    try: return json.loads(stdout)
    except: return None

def test(label, args, check_fn=None, timeout=120):
    stdout, stderr, rc, elapsed = run(args, timeout)
    data = parse_json(stdout)
    passed = False
    if check_fn:
        passed = check_fn(stdout, stderr, rc, data)
    else:
        passed = rc == 0 or (stdout and rc != -1)
    
    status = "PASS" if passed else "FAIL"
    detail = ""
    if isinstance(data, list) and data:
        detail = f" | {len(data)} results"
    if isinstance(data, dict):
        detail = f" | dict keys: {list(data.keys())[:5]}"
    if not passed and stderr:
        for line in stderr.split('\n'):
            l = line.strip()
            if ('Error' in l or 'error' in l or 'panic' in l) and len(l) < 120:
                detail += f" | {l}"
                break
    print(f"    [{status}] {label}: {elapsed:.2f}s{detail}")
    return passed

def get_preset_info(preset_name):
    """Get subsystem info for a preset"""
    stdout, _, rc, _ = run(["preset-info", "--preset", preset_name])
    data = parse_json(stdout)
    if isinstance(data, list) and data:
        for item in data:
            if item.get("preset") == preset_name:
                return item
    return {}

def test_preset(preset_name, subs, runtime):
    """Test features enabled for this preset"""
    data_dir = os.path.join(BASE_DATA, f"preset_test_{preset_name}")
    os.makedirs(data_dir, exist_ok=True)
    vec_path = os.path.join(data_dir, "vectors.bin")
    query_path = os.path.join(data_dir, "query.bin")
    gen_vectors(vec_path, N_VECTORS, DIM)
    gen_query(query_path, DIM)
    
    d = ["--data-dir", data_dir, "--dim", str(DIM)]
    results = []
    
    # --- Core (all presets) ---
    print("  --- Core ---")
    results.append(test("ingest-leader", d + ["ingest-leader", "--input", vec_path, "--target-clusters", "20"]))
    results.append(test("status", d + ["status"]))
    results.append(test("search-file", d + ["search-file", "--input", query_path, "--k", str(K)],
        check_fn=lambda so,se,rc,data: isinstance(data, list) and len(data) > 0))
    results.append(test("save", d + ["save"]))
    results.append(test("list", d + ["list"]))
    
    # --- Fused Search ---
    print("  --- Fused Search ---")
    results.append(test("fused-search", d + ["fused-search", "--query-file", query_path, "--k", str(K)],
        check_fn=lambda so,se,rc,data: isinstance(data, list) and len(data) > 0))
    
    # --- HNSW ---
    print("  --- HNSW ---")
    if runtime.get("has_hnsw"):
        results.append(test("hnsw-search", d + ["hnsw-search", "--query-file", query_path, "--k", str(K)],
            check_fn=lambda so,se,rc,data: isinstance(data, list) and len(data) > 0, timeout=120))
    else:
        print(f"    [SKIP] HNSW disabled")
        results.append(True)
    
    # --- LSH ---
    print("  --- LSH ---")
    if runtime.get("has_lsh"):
        results.append(test("lsh-search", d + ["lsh-search", "--query-file", query_path, "--k", str(K)],
            check_fn=lambda so,se,rc,data: isinstance(data, list) and len(data) > 0, timeout=120))
    else:
        print(f"    [SKIP] LSH disabled")
        results.append(True)
    
    # --- Quantization ---
    print("  --- Quantization ---")
    if runtime.get("has_quantization"):
        results.append(test("quant-index", d + ["quant-index", "--input", vec_path]))
        results.append(test("quant-search", d + ["quant-search", "-q", "1.0,0.5,-0.3", "-k", str(K)]))
        results.append(test("quant-status", d + ["quant-status"]))
    else:
        print(f"    [SKIP] Quantization disabled")
        results.append(True)
    
    # --- SOC ---
    print("  --- SOC ---")
    results.append(test("soc-check", d + ["soc-check"]))
    results.append(test("soc-relax", d + ["soc-relax", "--iterations", "5"]))
    
    # --- GPU Info ---
    print("  --- GPU ---")
    if subs.get("cuda") or subs.get("gpu_search"):
        results.append(test("gpu-info", d + ["gpu-info"]))
    else:
        print(f"    [SKIP] GPU not in preset")
        results.append(True)
    
    return results

def main():
    print("=" * 60)
    print(f"M2M Preset Validation — All Presets")
    print(f"Vectors: {N_VECTORS}x{DIM} | K: {K}")
    print("=" * 60)
    
    # 1. Gather preset info
    print("\n--- Preset Subsystem Map ---")
    preset_configs = {}
    for name in PRESETS:
        info = get_preset_info(name)
        subs = info.get("subsystems", {})
        runtime = info.get("runtime", {})
        preset_configs[name] = {"subs": subs, "runtime": runtime, "info": info}
        
        features = []
        for k, v in subs.items():
            if v: features.append(k)
        print(f"  {name:12s}: {', '.join(features) if features else '(core only)'}")
    
    # 2. Test each preset
    summary = {}
    for name in PRESETS:
        cfg = preset_configs[name]
        subs = cfg["subs"]
        runtime = cfg["runtime"]
        
        print(f"\n{'='*60}")
        print(f"PRESET: {name.upper()}")
        print(f"{'='*60}")
        
        results = test_preset(name, subs, runtime)
        p = sum(1 for r in results if r)
        f = sum(1 for r in results if not r)
        summary[name] = {"passed": p, "failed": f, "total": len(results)}
    
    # 3. Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    grand_p = 0
    grand_f = 0
    for name in PRESETS:
        s = summary[name]
        grand_p += s["passed"]
        grand_f += s["failed"]
        mark = "OK" if s["failed"] == 0 else f"{s['failed']} FAILED"
        print(f"  {name:12s}: {s['passed']:2d}/{s['total']:2d} passed | {mark}")
    
    print(f"\n  GRAND TOTAL: {grand_p} passed / {grand_f} failed / {grand_p+grand_f}")
    return 0 if grand_f == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
