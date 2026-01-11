"""Runtime manager service - Docker lifecycle management."""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import docker
import httpx
from docker.errors import DockerException, NotFound

from control_plane.config import settings

logger = logging.getLogger(__name__)


class RuntimeManager:
    """Manages Docker container lifecycle for agent runtimes."""

    def __init__(self):
        self._client: docker.DockerClient | None = None
        self._port_registry: dict[str, int] = {}

    @property
    def client(self) -> docker.DockerClient:
        """Get Docker client, initializing if needed."""
        if self._client is None:
            self._client = docker.DockerClient(base_url=settings.docker_host)
        return self._client

    def _allocate_port(self, establishment_id: str) -> int:
        """Allocate a port for a new runtime."""
        # Get all used ports from current containers
        used_ports = set()
        try:
            containers = self.client.containers.list(all=True)
            for container in containers:
                ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
                for port_bindings in ports.values():
                    if port_bindings:
                        for binding in port_bindings:
                            if binding.get("HostPort"):
                                used_ports.add(int(binding["HostPort"]))
        except DockerException as e:
            logger.warning(f"Could not list containers for port allocation: {e}")

        # Find an available port
        for port in range(settings.runtime_port_start, settings.runtime_port_end + 1):
            if port not in used_ports:
                self._port_registry[establishment_id] = port
                return port

        raise RuntimeError("No available ports in configured range")

    def generate_container_name(self, establishment_id: str) -> str:
        """Generate a unique container name for an establishment."""
        return f"agent_rt_{establishment_id}"

    async def create_runtime(
        self,
        establishment_id: str,
        image: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a new runtime container."""
        container_name = self.generate_container_name(establishment_id)
        image = image or settings.runtime_image
        host_port = self._allocate_port(establishment_id)

        # Prepare environment variables
        environment = {
            "ESTABLISHMENT_ID": establishment_id,
            "APP_PORT": "8080",
        }
        if env:
            environment.update(env)

        try:
            container = self.client.containers.create(
                image=image,
                name=container_name,
                detach=True,
                ports={"8080/tcp": host_port},
                environment=environment,
                network=settings.runtime_network,
                labels={
                    "control_plane": "true",
                    "establishment_id": establishment_id,
                },
            )

            return {
                "id": f"rt_{uuid4().hex[:12]}",
                "establishment_id": establishment_id,
                "container_name": container_name,
                "container_id": container.id,
                "image_tag": image,
                "host_port": host_port,
                "base_url": f"http://127.0.0.1:{host_port}",
                "status": "created",
            }
        except DockerException as e:
            logger.error(f"Failed to create runtime container: {e}")
            raise RuntimeError(f"Failed to create runtime: {e}")

    async def start_runtime(self, container_name: str) -> dict[str, Any]:
        """Start a runtime container."""
        try:
            container = self.client.containers.get(container_name)
            container.start()
            return {"status": "starting", "container_id": container.id}
        except NotFound:
            raise ValueError(f"Container {container_name} not found")
        except DockerException as e:
            logger.error(f"Failed to start container {container_name}: {e}")
            raise RuntimeError(f"Failed to start runtime: {e}")

    async def stop_runtime(self, container_name: str, timeout: int = 30) -> dict[str, Any]:
        """Stop a runtime container."""
        try:
            container = self.client.containers.get(container_name)
            container.stop(timeout=timeout)
            return {"status": "stopped", "container_id": container.id}
        except NotFound:
            raise ValueError(f"Container {container_name} not found")
        except DockerException as e:
            logger.error(f"Failed to stop container {container_name}: {e}")
            raise RuntimeError(f"Failed to stop runtime: {e}")

    async def restart_runtime(self, container_name: str, timeout: int = 30) -> dict[str, Any]:
        """Restart a runtime container."""
        try:
            container = self.client.containers.get(container_name)
            container.restart(timeout=timeout)
            return {"status": "running", "container_id": container.id}
        except NotFound:
            raise ValueError(f"Container {container_name} not found")
        except DockerException as e:
            logger.error(f"Failed to restart container {container_name}: {e}")
            raise RuntimeError(f"Failed to restart runtime: {e}")

    async def remove_runtime(self, container_name: str, force: bool = False) -> dict[str, Any]:
        """Remove a runtime container."""
        try:
            container = self.client.containers.get(container_name)
            container.remove(force=force)
            return {"status": "removed"}
        except NotFound:
            return {"status": "not_found"}
        except DockerException as e:
            logger.error(f"Failed to remove container {container_name}: {e}")
            raise RuntimeError(f"Failed to remove runtime: {e}")

    async def get_container_status(self, container_name: str) -> dict[str, Any]:
        """Get container status."""
        try:
            container = self.client.containers.get(container_name)
            return {
                "container_id": container.id,
                "status": container.status,
                "running": container.status == "running",
            }
        except NotFound:
            return {"status": "not_found", "running": False}
        except DockerException as e:
            logger.error(f"Failed to get container status: {e}")
            return {"status": "error", "running": False, "error": str(e)}

    async def check_runtime_health(
        self, base_url: str, timeout: float = 5.0
    ) -> dict[str, Any]:
        """Check runtime health via /readyz endpoint."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base_url}/readyz")
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "ready": data.get("ready", True),
                        "issues": data.get("issues", []),
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    return {
                        "ready": False,
                        "issues": [f"Health check returned {response.status_code}"],
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                    }
        except httpx.RequestError as e:
            return {
                "ready": False,
                "issues": [f"Health check failed: {e}"],
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

    async def get_runtime_info(self, base_url: str, timeout: float = 5.0) -> dict[str, Any] | None:
        """Get runtime info via /info endpoint."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base_url}/info")
                if response.status_code == 200:
                    return response.json()
        except httpx.RequestError as e:
            logger.warning(f"Failed to get runtime info from {base_url}: {e}")
        return None

    async def drain_runtime(self, base_url: str, timeout: float = 5.0) -> dict[str, Any]:
        """Send drain command to runtime."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{base_url}/control/drain")
                return {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                }
        except httpx.RequestError as e:
            return {"success": False, "error": str(e)}

    async def resume_runtime(self, base_url: str, timeout: float = 5.0) -> dict[str, Any]:
        """Send resume command to runtime."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{base_url}/control/resume")
                return {
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                }
        except httpx.RequestError as e:
            return {"success": False, "error": str(e)}


# Singleton instance
runtime_manager = RuntimeManager()
