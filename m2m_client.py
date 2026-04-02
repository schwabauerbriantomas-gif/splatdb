"""
M2M Vector Search - Python Client

Usage:
    from m2m_client import M2MClient

    client = M2MClient()
    client.store("My important memory", category="notes")
    results = client.search("important", top_k=5)
    print(results)
"""

import json
import subprocess
import time
import signal
import sys
import requests
from typing import Optional, List, Dict, Any


class M2MClient:
    """Client for M2M Vector Search via HTTP API."""

    def __init__(self, port: int = 8199, host: str = "127.0.0.1",
                 auto_start: bool = True, binary_path: Optional[str] = None):
        self.port = port
        self.host = host
        self.base_url = f"http://{host}:{port}"
        self._process = None
        self._binary_path = binary_path

        if auto_start and not self._is_running():
            self.start_server()

    def _is_running(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=1)
            return r.status_code == 200
        except Exception:
            return False

    def start_server(self, binary_path: Optional[str] = None):
        """Start the M2M server as a subprocess."""
        path = binary_path or self._binary_path
        if path is None:
            # Try to find the binary
            import shutil
            path = shutil.which("m2m-vector-search")
            if path is None:
                # Check common locations
                candidates = [
                    "D:/m2m-memory/target/release/m2m-vector-search.exe",
                    "D:/m2m-memory/target/debug/m2m-vector-search.exe",
                ]
                for c in candidates:
                    import os
                    if os.path.exists(c):
                        path = c
                        break
            if path is None:
                raise FileNotFoundError(
                    "Cannot find m2m-vector-search binary. "
                    "Build with: cargo build --release --features cuda"
                )

        self._process = subprocess.Popen(
            [path, "serve", "--port", str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        # Wait for server to be ready
        for _ in range(30):
            if self._is_running():
                return
            time.sleep(0.5)
        raise RuntimeError("Server failed to start within 15 seconds")

    def stop_server(self):
        """Stop the M2M server subprocess."""
        if self._process:
            if sys.platform == "win32":
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None

    def store(self, text: str, category: Optional[str] = None,
              id: Optional[str] = None, embedding: Optional[List[float]] = None) -> Dict:
        """Store a memory."""
        payload = {"text": text}
        if category:
            payload["category"] = category
        if id:
            payload["id"] = id
        if embedding:
            payload["embedding"] = embedding

        r = requests.post(f"{self.base_url}/store", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()

    def search(self, query: str, top_k: int = 10,
               embedding: Optional[List[float]] = None) -> List[Dict]:
        """Search memories by query text."""
        payload = {"query": query, "top_k": top_k}
        if embedding:
            payload["embedding"] = embedding

        r = requests.post(f"{self.base_url}/search", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()["results"]

    def status(self) -> Dict:
        """Get store status."""
        r = requests.post(f"{self.base_url}/status", timeout=5)
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict:
        """Health check."""
        r = requests.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop_server()

    def __del__(self):
        self.stop_server()


if __name__ == "__main__":
    # Quick smoke test
    print("M2M Python Client - Smoke Test")
    print("=" * 40)

    try:
        client = M2MClient(auto_start=True)
        print(f"Server health: {client.health()}")

        # Store some memories
        client.store("Gaussian splatting for neural rendering", category="research")
        client.store("Energy-based models for density estimation", category="research")
        client.store("I need to buy groceries tomorrow", category="todo")
        client.store("Meeting with team at 3pm", category="calendar")

        print(f"\nStatus: {client.status()}")

        # Search
        results = client.search("machine learning rendering")
        print(f"\nSearch 'machine learning rendering': {len(results)} results")
        for r in results:
            print(f"  idx={r['index']} score={r['score']:.4f}")

        results = client.search("tasks and plans")
        print(f"\nSearch 'tasks and plans': {len(results)} results")
        for r in results:
            print(f"  idx={r['index']} score={r['score']:.4f}")

        client.stop_server()
        print("\nDone!")
    except Exception as e:
        print(f"Error: {e}")
