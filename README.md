# Pathology MCP Server

A Model Context Protocol (MCP) server for pathology workflows and AI agent integration.

## Overview
This project provides tools for pathology data access, analysis, and integration with agentic AI systems.

## Tools

The MCP server exposes the following tools to AI agents:

- **create_patches** – Extract image patches from a whole-slide image
- **compute_embeddings** – Generate feature embeddings for patches  
  (Supported models: Path Foundation, Virchow, UNI2)
- **cluster_embeddings** – Cluster patch embeddings
- **find_similar_patches** – Retrieve visually similar patches
- **run_qc** – Perform slide quality control checks

## Installation

```bash
git clone https://github.com/Navoditamathur/pathology-copilot.git
cd pathology-copilot
pip install -r requirements.txt
python src/server.py run # to run in http
edit mcp config to this path and allow cloud to run
```


## Example Query
Example instruction for an AI agent:
"Create patches from the slide:
pathology-copilot/input/SR386_40X_HE_T049_01.czi
and store them in:
pathology-copilot/patches
using 40x magnification."
