#!/usr/bin/env python3
"""
Dashboard Integration Test Script

Tests the complete dashboard functionality:
1. Backend health check
2. Config endpoint
3. File upload (multimodal ingestion)
4. Ingestion status polling
5. Database stats
6. Retrieval query

Usage:
    # Start backend first:
    cd dashboard/backend && uvicorn main:app --reload --port 8000
    
    # Then run test:
    uv run python scripts/dashboard_test.py
"""

import asyncio
import httpx
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

API_BASE = "http://localhost:8009/api"
TEST_FILES_DIR = Path(__file__).parent.parent / "data"


async def test_health() -> bool:
    """Test health endpoint."""
    console.print("\n[bold cyan]1. Health Check[/bold cyan]")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE}/health", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                console.print(f"   [green]✓[/green] Backend healthy: {data}")
                return True
            else:
                console.print(f"   [red]✗[/red] Health check failed: {response.status_code}")
                return False
        except Exception as e:
            console.print(f"   [red]✗[/red] Cannot connect to backend: {e}")
            console.print(f"   [dim]Make sure backend is running: uvicorn main:app --reload --port 8000[/dim]")
            return False


async def test_config() -> bool:
    """Test config endpoint."""
    console.print("\n[bold cyan]2. Configuration[/bold cyan]")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE}/config", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                categories = data.get("categories", [])
                config_count = len(data.get("config", {}))
                
                console.print(f"   [green]✓[/green] Loaded {config_count} config items")
                console.print(f"   [dim]Categories: {', '.join(categories[:5])}...[/dim]")
                return True
            else:
                console.print(f"   [red]✗[/red] Config failed: {response.status_code}")
                return False
        except Exception as e:
            console.print(f"   [red]✗[/red] Config error: {e}")
            return False


async def test_database_stats() -> bool:
    """Test database stats endpoint."""
    console.print("\n[bold cyan]3. Database Stats[/bold cyan]")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE}/database/stats", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                
                table = Table(show_header=True, header_style="bold")
                table.add_column("Metric")
                table.add_column("Count", justify="right")
                
                for key, value in data.items():
                    if key != "error":
                        table.add_row(key.replace("_", " ").title(), str(value))
                
                console.print(table)
                
                if "error" in data:
                    console.print(f"   [yellow]⚠[/yellow] DB warning: {data['error']}")
                
                return True
            else:
                console.print(f"   [red]✗[/red] Stats failed: {response.status_code}")
                return False
        except Exception as e:
            console.print(f"   [red]✗[/red] Stats error: {e}")
            return False


async def test_file_upload(file_path: Path) -> str | None:
    """Test file upload and return job ID."""
    console.print(f"\n[bold cyan]4. File Upload: {file_path.name}[/bold cyan]")
    
    if not file_path.exists():
        console.print(f"   [red]✗[/red] Test file not found: {file_path}")
        return None
    
    async with httpx.AsyncClient() as client:
        try:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/octet-stream")}
                response = await client.post(
                    f"{API_BASE}/ingest/upload",
                    files=files,
                    timeout=30.0
                )
            
            if response.status_code == 200:
                data = response.json()
                job_id = data.get("job_id")
                file_type = data.get("file_type")
                
                console.print(f"   [green]✓[/green] Upload started")
                console.print(f"   [dim]Job ID: {job_id}[/dim]")
                console.print(f"   [dim]File Type: {file_type}[/dim]")
                
                return job_id
            else:
                error = response.json().get("detail", response.text)
                console.print(f"   [red]✗[/red] Upload failed: {error}")
                return None
        except Exception as e:
            console.print(f"   [red]✗[/red] Upload error: {e}")
            return None


async def test_ingestion_status(job_id: str, max_wait: int = 120) -> bool:
    """Poll ingestion status until complete."""
    console.print(f"\n[bold cyan]5. Ingestion Progress[/bold cyan]")
    
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        last_progress = -1
        
        while time.time() - start_time < max_wait:
            try:
                response = await client.get(
                    f"{API_BASE}/ingest/status/{job_id}",
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    console.print(f"   [red]✗[/red] Status check failed: {response.status_code}")
                    return False
                
                data = response.json()
                status = data.get("status")
                progress = data.get("progress", 0)
                stages = data.get("stages", [])
                
                # Print progress update if changed
                progress_pct = int(progress * 100)
                if progress_pct != last_progress:
                    last_progress = progress_pct
                    
                    # Show current stage
                    current_stage = next((s for s in stages if s["status"] == "running"), None)
                    stage_name = current_stage["name"] if current_stage else "..."
                    
                    console.print(f"   [{progress_pct:3d}%] {stage_name}")
                
                if status == "completed":
                    result = data.get("result", {})
                    console.print(f"   [green]✓[/green] Ingestion completed!")
                    
                    # Show result summary
                    table = Table(show_header=False, box=None, padding=(0, 2))
                    table.add_column("Metric")
                    table.add_column("Value", justify="right")
                    
                    table.add_row("Pages", str(result.get("pages", 0)))
                    table.add_row("OCR Pages", str(result.get("pages_ocr", 0)))
                    table.add_row("Parent Chunks", str(result.get("parent_chunks", 0)))
                    table.add_row("Child Chunks", str(result.get("child_chunks", 0)))
                    table.add_row("Entities", str(result.get("entities", 0)))
                    
                    console.print(table)
                    return True
                
                elif status == "failed":
                    error = data.get("error", "Unknown error")
                    console.print(f"   [red]✗[/red] Ingestion failed: {error}")
                    
                    # Show failed stage
                    for stage in stages:
                        if stage["status"] == "failed":
                            console.print(f"   [dim]Failed stage: {stage['name']}[/dim]")
                            if stage.get("error"):
                                console.print(f"   [dim]Error: {stage['error']}[/dim]")
                    
                    return False
                
                await asyncio.sleep(1)
                
            except Exception as e:
                console.print(f"   [red]✗[/red] Status error: {e}")
                await asyncio.sleep(2)
        
        console.print(f"   [yellow]⚠[/yellow] Timeout waiting for ingestion")
        return False


async def test_retrieval(query: str = "What is the document about?") -> bool:
    """Test retrieval endpoint."""
    console.print(f"\n[bold cyan]6. Retrieval Query[/bold cyan]")
    console.print(f"   [dim]Query: {query}[/dim]")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_BASE}/retrieve",
                json={
                    "query": query,
                    "tenant_id": "default",
                    "top_k": 5
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                channels = data.get("channels", {})
                
                console.print(f"   [green]✓[/green] Retrieval completed")
                console.print(f"   [dim]Results: {len(results)}[/dim]")
                console.print(f"   [dim]Channels - Semantic: {channels.get('semantic', 0)}, "
                            f"Lexical: {channels.get('lexical', 0)}, "
                            f"Graph: {channels.get('graph', 0)}[/dim]")
                
                # Show top results
                if results:
                    console.print("\n   Top Results:")
                    for i, result in enumerate(results[:3]):
                        text = result.get("text", "")[:100].replace("\n", " ")
                        score = result.get("final_score", 0)
                        console.print(f"   [{i+1}] score={score:.4f}")
                        console.print(f"       {text}...")
                
                return True
            else:
                error = response.json().get("detail", response.text)
                console.print(f"   [red]✗[/red] Retrieval failed: {error}")
                return False
        except Exception as e:
            import traceback
            console.print(f"   [red]✗[/red] Retrieval error: {e}")
            traceback.print_exc()
            return False


async def test_metrics() -> bool:
    """Test metrics endpoint."""
    console.print(f"\n[bold cyan]7. Metrics[/bold cyan]")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE}/metrics", timeout=10.0)
            
            if response.status_code == 200:
                data = response.json()
                
                # Config status
                config = data.get("config", {})
                enabled_features = [k.replace("_enabled", "") for k, v in config.items() if v is True]
                console.print(f"   [green]✓[/green] Enabled features: {', '.join(enabled_features)}")
                
                # Ingestion jobs
                jobs = data.get("ingestion_jobs", {})
                console.print(f"   [dim]Jobs - Total: {jobs.get('total', 0)}, "
                            f"Completed: {jobs.get('completed', 0)}, "
                            f"Failed: {jobs.get('failed', 0)}[/dim]")
                
                return True
            else:
                console.print(f"   [red]✗[/red] Metrics failed: {response.status_code}")
                return False
        except Exception as e:
            console.print(f"   [red]✗[/red] Metrics error: {e}")
            return False


def find_test_file() -> Path | None:
    """Find a suitable test file."""
    # Priority order: small files first
    extensions = [".txt", ".md", ".csv", ".docx", ".xlsx", ".pdf", ".png", ".jpg"]
    
    for ext in extensions:
        for f in TEST_FILES_DIR.iterdir():
            if f.suffix.lower() == ext and f.stat().st_size < 10_000_000:  # < 10MB
                return f
    
    return None


async def main():
    """Run all dashboard tests."""
    console.print(Panel.fit(
        "[bold cyan]Triple-Hybrid-RAG Dashboard Test[/bold cyan]\n"
        "[dim]Testing backend API endpoints[/dim]",
        border_style="cyan"
    ))
    
    results = {}
    
    # 1. Health check
    results["health"] = await test_health()
    if not results["health"]:
        console.print("\n[red]Backend not running. Exiting.[/red]")
        return 1
    
    # 2. Config
    results["config"] = await test_config()
    
    # 3. Database stats
    results["db_stats"] = await test_database_stats()
    
    # 4-5. File upload & ingestion
    test_file = find_test_file()
    if test_file:
        job_id = await test_file_upload(test_file)
        if job_id:
            results["upload"] = True
            results["ingestion"] = await test_ingestion_status(job_id)
        else:
            results["upload"] = False
            results["ingestion"] = False
    else:
        console.print("\n[yellow]⚠[/yellow] No test files found in data/ directory")
        results["upload"] = None
        results["ingestion"] = None
    
    # 6. Retrieval
    results["retrieval"] = await test_retrieval()
    
    # 7. Metrics
    results["metrics"] = await test_metrics()
    
    # Summary
    console.print("\n" + "=" * 50)
    console.print("[bold]Test Summary[/bold]")
    console.print("=" * 50)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        if result is True:
            status = "[green]✓ PASS[/green]"
        elif result is False:
            status = "[red]✗ FAIL[/red]"
        else:
            status = "[yellow]⊘ SKIP[/yellow]"
        
        console.print(f"  {test_name:15s} {status}")
    
    console.print(f"\n[bold]Result: {passed} passed, {failed} failed, {skipped} skipped[/bold]")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
