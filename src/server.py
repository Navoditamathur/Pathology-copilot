#!/usr/bin/env python3
"""
Pathology MCP Server - Slide-based MSI prediction and quality control system.
"""

import os
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from eliot import start_action
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from core import PathologyEngine
from schemas import *


# =========================
# ===== ENV SETUP =========
# =========================

load_dotenv()

DEFAULT_HOST = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MCP_PORT", "3001"))
DEFAULT_TRANSPORT = os.getenv("MCP_TRANSPORT", "http")


# =========================
# ===== MCP SERVER =========
# =========================

class PathologyMCP(FastMCP):
    """
    Pathology MCP Server with slide analysis tools.
    """

    def __init__(
        self,
        name: str = "Pathology MCP Server",
        config: PathologyConfig|None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.core = PathologyEngine(config)
        self._register_tools()

    # =========================
    # ===== TOOL REGISTRATION ==
    # =========================

    def _register_tools(self):

        @self.tool(
            name="create_patches",
            description="""
            Extract image patches from a whole-slide image.
            Supports local WSI or DICOMweb slide references.
            """
        )
        def create_patches(request: SlideInput) -> dict[str, Any]:
            with start_action(action_type="create_patches"):
                result = self.core.generate_patches(
                    mode='local',
                    patch_dir=request.patch_dir,
                    input_path=request.input_path,
                    study_uid=request.study_uid,
                    series_uid=request.series_uid,
                    magnification=request.magnification)


                return PatchCreateResponse(result.dict())

        @self.tool(
            name=f"compute_embeddings",
            description="""
            Computes feature embeddings for slide patches.
            """
        )
        def compute_embeddings(request: EmbeddingRequest) -> dict[str, Any]:
            with start_action(action_type="compute_embeddings"):
                result = self.core.create_embeddings(
                    patch_dir=request.patch_dir,
                    model_name=request.model_name,
                )
                return result.dict()

        @self.tool(
            name=f"cluster_embeddings",
            description="""
            Creates and clusters feature embeddings for slide patches.
            """
        )
        def cluster_embeddings(request: EmbeddingRequest) -> dict[str, Any]:
            with start_action(action_type="cluster_embeddings"):
                result = self.core.cluster_embeddings(
                    patch_dir=request.patch_dir,
                    model_name=request.model_name,
                )
                return result.dict()

        @self.tool(
            name="find_similar_patches",
            description="""
            Retrieve visually similar patches using embedding similarity.
            """
        )
        def find_similar_patches(request: SimilarPatchRequest):
            with start_action(action_type="find_similar_patches"):
                result = self.core.find_similar_patches(
                    patch_dir=request.patch_dir,
                    query_patch=request.query_patch,
                    top_k=request.top_k,
                    model_name=request.model_name,
                )
                return result.dict()

        @self.tool(
            name="run_qc",
            description="""
            Perform slide-level quality control:
            - Tissue coverage
            - Blur detection
            - Stain quality assessment
            """
        )
        def run_qc(request: QCRequest):
            with start_action(action_type="run_qc"):
                result = self.core.run_slide_quality_control(
                    patch_dir=request.patch_dir
                )
                return result.dict()

        @self.tool(
            name="get_server_metadata",
            description="Return server configuration and capabilities."
        )
        def get_server_metadata() -> dict[str, Any]:
            return {
                "server_name": self.name,
                "tools": [
                    "create_patches",
                    "compute_embeddings",
                    "cluster_embeddings",
                    "find_similar_patches",
                    "run_qc",
                ],
            }


# =========================
# ===== CLI APP ============
# =========================

app = typer.Typer(
    help="Pathology MCP Server - Slide-based MSI prediction system"
)


@app.command("run")
def run_server(
    host: str = typer.Option(DEFAULT_HOST),
    port: int = typer.Option(DEFAULT_PORT),
    transport: str = typer.Option(DEFAULT_TRANSPORT),
):
    mcp = PathologyMCP()
    mcp.run(transport=transport, host=host, port=port)


@app.command("stdio")
def run_stdio():
    mcp = PathologyMCP()
    mcp.run()


@app.command("sse")
def run_sse(
    host: str = typer.Option(DEFAULT_HOST),
    port: int = typer.Option(DEFAULT_PORT),
):
    mcp = PathologyMCP()
    mcp.run()


# =========================
# ===== ENTRY POINT ========
# =========================

if __name__ == "__main__":
    import sys

    # If no CLI arguments are passed, assume MCP stdio mode (Claude)
    if len(sys.argv) == 1:
        mcp = PathologyMCP()
        mcp.run()  # stdio
    else:
        app()