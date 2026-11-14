# Pathology Copilot

An LLM can't read a whole-slide image (WSI). A single H&E slide is billions of pixels, far beyond any model's context window. Pathology Copilot is a Model Context Protocol (MCP) server that gives an LLM agent tools to work with a slide indirectly: it cuts the slide into patches, checks their quality, embeds them with pathology foundation models, and finds representative or similar regions. The agent then reasons over small, structured results instead of raw pixels.

## Tools

1. **`create_patches`** reads the slide at the requested magnification, picks the pyramid level that matches it, and saves up to 500 randomly sampled 224×224 patches (fixed seed). Sources can be local OpenSlide formats, Zeiss `.czi`, or DICOMweb (study/series UIDs).
2. **`run_qc`** scores the patches for tissue coverage (grey-level threshold), blur (Laplacian variance), and stain quality (HSV saturation), then returns an overall `high / acceptable / low` rating.
3. **`compute_embeddings`** embeds each patch with a pathology foundation model and caches the vectors on disk, one folder per model.
4. **`cluster_embeddings`** runs k-means over the embeddings and returns the medoid patch of each cluster. This gives the agent a handful of patches that represent the slide's main tissue types.
5. **`find_similar_patches`** ranks patches by cosine similarity to a query patch.


## Limitations and next steps

- There is no cohort-level indexing or cross-slide search.
- Requires a GPU, and Hugging Face access approval for the gated models.
- Plan to add tool for PHI Handling to limit exposure of sensitive data to models. 


## Setup

```bash
git clone https://github.com/Navoditamathur/Pathology-copilot.git
cd Pathology-copilot
pip install -r requirements.txt
cp .env.example .env        # add your HF_TOKEN (needed for gated models)
```

### Connect to Claude Desktop (stdio)

Add this to `claude_desktop_config.json` and restart Claude Desktop:

```json
{
  "mcpServers": {
    "pathology-copilot": {
      "command": "python",
      "args": ["/absolute/path/to/Pathology-copilot/src/server.py"],
      "env": { "HF_TOKEN": "your-token" }
    }
  }
}
```

Running `server.py`. Claude Desktop uses stdio mode.

### Run over HTTP

```bash
python src/server.py run --host 127.0.0.1 --port 3001
```

Then point your MCP client at `http://127.0.0.1:3001/mcp`.

## Example

> "Create patches from `input/SR386_40X_HE_T049_01.czi` at 40x, run QC, then cluster them with Path Foundation and show me the representative patch for each cluster."
