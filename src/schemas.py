from pydantic import BaseModel, Field
from typing import Optional, Literal, List

class PathologyConfig(BaseModel):
    base_dicomweb: str = "https://healthcare.googleapis.com/v1/projects/hai-cd3-foundations/locations/us-central1/datasets/public/dicomStores/test-images/dicomWeb"
    inputdir : str = "/input/"
    patch_root: str = "/patches/"
    patch_size: int = 224
    seed: int = 42
    max_patches: int = 500
    embedding_model: Literal["google/path-foundation", "vit_giant_patch14_224", "hf-hub:paige-ai/Virchow2"] = "google/path-foundation"

class SlideInput(BaseModel):
    mode: Literal["local", "dicomweb"] = "local"
    patch_dir: str = Field(default="/patches/")
    magnification: str = "10X"

    # For local
    input_path: Optional[str] = None

    # For DICOMweb
    study_uid: Optional[str] = None
    series_uid: Optional[str] = None

class PatchCreateResponse(BaseModel):
    patch_dir: str = Field(default="/patches/")
    num_patches: int = Field(default=6)

class EmbeddingRequest(BaseModel):
    patch_dir: str = Field(default="/patches/")
    model_name: Optional[str] = "google/path-foundation"
    num_clusters: Optional[int] = 6

class ClusterSchema(BaseModel):
    cluster_id: int 
    medoid_patch: str
    patch_count: int

class ClusterResult(BaseModel):
    clusters: List[ClusterSchema]

class SimilarPatchRequest(BaseModel):
    patch_dir: str
    query_patch: str
    top_k: int
    model_name: str

class SimilarPatchSchema(BaseModel):
    patch_name: str
    similarity_score: float
    
class SimilarPatchResult(BaseModel):
    patches: List[SimilarPatchSchema]

class QCRequest(BaseModel):
    patch_dir : str

class QCResult(BaseModel):
    tissue_coverage_percent: float
    blur_fraction: float
    staining_quality: str
    overall_quality: str
