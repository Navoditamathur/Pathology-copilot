
#eg Use tool create_patches to create patches of /Users/navoditamathur/Desktop/pathology-copilot/input/SR386_40X_HE_T049_01.czi in /Users/navoditamathur/Desktop/pathology-copilot/patches avaiable at 40x
from schemas import PathologyConfig
import openslide
from models import EmbeddingModel
import numpy as np
from aicspylibczi import CziFile
from PIL import Image
from pathlib import Path
import os
from aicsimageio import AICSImage


class PathologyEngine:
    def __init__(self, config=None):
        if config is None:
            self.config = PathologyConfig()  # use defaults
        elif isinstance(config, PathologyConfig):
            self.config = config
        else:
            self.config = PathologyConfig.model_validate(config)

    def _load_model(self):
        print("Loading MedGemma...")
        self.processor = transformers.AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        print("Model loaded.")

    def _get_embedings_path(patch_dir: str, model_name="google/path-foundation"):
        model_safe_name = model_name.replace("/", "_")
        model_dir = patch_dir / model_safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def _get_slide(study_uid: str, series_uid: str):
        series_url = f"{self.base_dicomweb}/studies/{study_uid}/series/{series_uid}"
        dwi = dicom_web_interface.DicomWebInterface(
            credential_factory.DefaultCredentialFactory()
        )
        slide = dicom_slide.DicomSlide(dwi, dicom_path.FromString(series_url))
        slide.init_slide_frame_cache()
        return slide

    def _load_czi_rgb(self, path):
        czi = CziFile(path)

        # Read full image
        image, dims = czi.read_image()

        # Remove extra dims
        # Typical dims: (1, 1, 1, Y, X, 3)
        image = np.squeeze(image)

        # If channels first, move them
        if image.shape[0] in [3, 4] and image.ndim == 3:
            image = np.moveaxis(image, 0, -1)

        return image


    def _generate_patches_czi(self, slide_path: str, output_dir: str):

        img = AICSImage(slide_path)
        data = img.get_image_data("TCZYX")

        height = data.shape[-2]
        width = data.shape[-1]

        patch_size = self.config.patch_size

        output_dir = Path(output_dir) / Path(slide_path).stem
        os.makedirs(output_dir, exist_ok=True)

        coords = []
        for y in range(0, height - patch_size, self.config.stride):
            for x in range(0, width - patch_size, self.config.stride):
                coords.append((x, y))

        coords = random.sample(coords, k=min(self.config.max_patches, len(coords)))

        patch_count = 0

        for i, (x, y) in enumerate(coords):

            patch = data[
                0,        # T
                :,        # C
                0,        # Z
                y:y+patch_size,
                x:x+patch_size
            ]

            patch = np.moveaxis(patch, 0, -1)

            img_patch = Image.fromarray(patch.astype(np.uint8))
            img_patch.save(f"{output_dir}/patch_{i}.jpg")

            patch_count += 1

        return {
            "patch_dir": str(output_dir),
            "patch_count": patch_count
        }

    def _create_patches_from_local(
        self,
        input_path: str,
        patch_dir: str,
        magnification: str = "40X",
    ):
        """
        Extract patches from a local whole-slide image using OpenSlide.
        Deterministic and magnification-aware.
        """

        slide_path = Path(input_path)

        if not slide_path.exists():
            raise FileNotFoundError(f"Slide not found: {input_path}")

        if input_path.endswith('.czi'):
            return self._generate_patches_czi(input_path, patch_dir)

        slide = openslide.OpenSlide(str(slide_path))
        objective_power = float(
            slide.properties.get("openslide.objective-power", 20)
        )
        target_mag = float(magnification.replace("X", ""))
        downsample = objective_power / target_mag
        level = slide.get_best_level_for_downsample(downsample)
        level_downsample = slide.level_downsamples[level]
        level_dim = slide.level_dimensions[level]
        random.seed(self.config.seed)

        output_dir = Path(patch_dir) / input_path.name 
        os.makedirs(output_dir, exist_ok=True)

        width, height = level_dim
        coords = []
        for y in range(0, height - self.config.patch_size, self.config.stride):
            for x in range(0, width - self.config.patch_size, self.config.stride):
                coords.append((x, y))

        if len(coords) == 0:
            raise ValueError("Slide too small for given patch size.")

        coords = random.sample(coords, k=min(self.config.max_patches, len(coords)))
        patch_count = 0
        for i, (x, y) in enumerate(coords):
            region = slide.read_region(
                location=(
                    int(x * level_downsample),
                    int(y * level_downsample),
                ),
                level=level,
                size=(self.config.patch_size, self.config.patch_size),
            )

            region = region.convert("RGB")
            region.save(output_dir / f"patch_{i}.jpg")
            patch_count += 1

        slide.close()

        return {
            "patch_dir": str(output_dir),
            "patch_count": patch_count
        }

    def _generate_patches_from_dicomweb(self, patch_dir: str, study_uid: str, series_uid: str, magnification: str):
        if not self.study_uid or not self.series_uid:
            raise ValueError("Study UID and series UID are required for dicomweb mode")
        slide = self.get_slide(study_uid, series_uid)
        level = slide.get_level_by_pixel_spacing(
            pixel_spacing.PixelSpacing.FromMagnificationString(magnification)
        )
        if level is None:
            raise ValueError(f"No level found for magnification {magnification}")

        generator = patch_generator.DicomPatchGenerator(
            slide,
            level,
            patch_size=self.config.patch_size,
            stride_size=self.config.patch_size,
        )
        patches = [p for p in generator]
        patches = random.sample(patches, k=min(self.config.max_patches, len(patches)))

        output_dir = Path(patch_dir) / magnification /f"{study_uid}_{series_uid}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, p in enumerate(patches):
            img = Image.fromarray(p.image_bytes())
            img.save(output_dir / f"patch_{i}.jpg")

        return {
            "patch_dir": str(output_dir),
            "patch_count": len(patches)
        }

    def generate_patches(self, mode: str, patch_dir: str, input_path: str, study_uid: str, series_uid: str, magnification: str):
        if mode == 'local':
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"File {input_path} not found")
            patches = self._create_patches_from_local(input_path, patch_dir, magnification)
        elif mode == 'dicomweb':
            if not self.study_uid or not self.series_uid:
                raise ValueError("Study UID and series UID are required for dicomweb mode")
            patches = self._generate_patches_from_dicomweb(patch_dir, study_uid, series_uid, magnification)
            return patches
        else:
            raise ValueError("Invalid mode")

    def _compute_tissue_coverage(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        tissue_pixels = np.sum(gray < self.tissue_threshold)
        total_pixels = gray.size

        return tissue_pixels / total_pixels

    # -------------------------
    # Blur Detection
    # -------------------------
    def _compute_blur_score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    # -------------------------
    # Stain Quality
    # -------------------------
    def _compute_stain_quality(self, image: np.ndarray) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        return mean_saturation

    def run_slide_quality_control(patch_dir: str, 
        tissue_threshold: int = 220,
        blur_threshold: float = 80.0,
        saturation_good: float = 50.0,
        saturation_moderate: float = 30.0,) -> dict:

        patch_dir = Path(patch_dir)

        if not patch_dir.exists():
            raise ValueError(f"Patch directory does not exist: {patch_dir}")

        image_paths = list(patch_dir.glob("*.jpg")) + list(patch_dir.glob("*.png"))

        if len(image_paths) == 0:
            raise ValueError("No image patches found in directory.")

        tissue_coverages: List[float] = []
        blur_flags: List[bool] = []
        stain_scores: List[float] = []

        for img_path in image_paths:
            image = cv2.imread(str(img_path))

            if image is None:
                continue

            # Tissue
            tissue_cov = self._compute_tissue_coverage(image)
            tissue_coverages.append(tissue_cov)

            # Blur
            blur_score = self._compute_blur_score(image)
            blur_flags.append(blur_score < self.blur_threshold)

            # Stain
            stain_score = self._compute_stain_quality(image)
            stain_scores.append(stain_score)

        # Aggregate
        avg_tissue = float(np.mean(tissue_coverages))
        blur_fraction = float(np.mean(blur_flags))
        avg_saturation = float(np.mean(stain_scores))

        # Interpret staining quality
        if avg_saturation > self.saturation_good:
            staining_quality = "good"
        elif avg_saturation > self.saturation_moderate:
            staining_quality = "moderate"
        else:
            staining_quality = "poor"

        # Overall quality logic
        overall_quality = "high"

        if avg_tissue < 0.3:
            overall_quality = "low"
        elif blur_fraction > 0.3:
            overall_quality = "low"
        elif staining_quality == "poor":
            overall_quality = "low"
        elif avg_tissue < 0.6:
            overall_quality = "acceptable"

        return {
            tissue_coverage_percent:round(avg_tissue * 100, 2),
            blur_fraction:round(blur_fraction, 3),
            staining_quality:staining_quality,
            overall_quality:overall_quality,
        }

    def create_embeddings(patch_dir: str, model_name="google/path-foundation"):
        patch_dir = Path(patch_dir)
        embeddings_path = get_embedings_path(patch_dir, model_name)
        if (embeddings_path / "embeddings.npy").exists() and (embeddings_path / "filenames.json").exists():
            return {
                "embedding_file": str(embeddings_path / "embeddings.npy"),
                "filenames_file": str(embeddings_path / "filenames.json")
            }
        image_files = sorted(patch_dir.glob("*.jpg"))

        # Download the entire model repository
        embeddings = []
        filenames = []

        model = EmbeddingModel(model_name, device=self.device)

        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            emb = model.embed(img)
            embeddings.append(emb)
            filenames.append(img_path.name)

        embeddings = np.array(embeddings)

        np.save(embeddings_path / "embeddings.npy", embeddings)

        with open(embeddings_path / "filenames.json", "w") as f:
            json.dump(filenames, f)

        return {
            "embedding_file": str(embeddings_path / "embeddings.npy"),
            "filenames_file": str(embeddings_path / "filenames.json")
        }

    def cluster_embeddings(patch_dir: str, model_name="google/path-foundation", num_clusters: int = 6):
        patch_dir = Path(patch_dir)
        embeddings_path = self._get_embedings_path(patch_dir, model_name)

        if not (embeddings_path / "embeddings.npy").exists():
            self.create_embeddings(patch_dir, model_name)

        embeddings = np.load(embeddings_path / "embeddings.npy")

        with open(embeddings_path / "filenames.json") as f:
            filenames = json.load(f)

        if len(embeddings) == 0:
            return {"error": "No embeddings found"}

        kmeans = KMeans(n_clusters=min(num_clusters, len(embeddings)), random_state=42)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        clusters_output = []

        for cluster_id in range(len(centroids)):
            indices = np.where(labels == cluster_id)[0]
            cluster_embeddings = embeddings[indices]

            centroid = centroids[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            medoid_idx = indices[np.argmin(distances)]

            clusters_output.append({
                "cluster_id": int(cluster_id),
                "medoid_patch": filenames[medoid_idx],
                "patch_count": int(len(indices))
            })

        return clusters_output

    def retrieve_similar_patches(patch_dir: str, query_patch: str, top_k: int = 5, model_name="google/path-foundation"):
        patch_dir = Path(patch_dir)
        embeddings_path = self._get_embedings_path(patch_dir, model_name)

        if not (embeddings_path / "embeddings.npy").exists():
            self.create_embeddings(patch_dir, model_name)

        embeddings = np.load(embeddings_path / "embeddings.npy")
        with open(embeddings_path / "filenames.json") as f:
            filenames = json.load(f)

        if query_patch not in filenames:
            return {"error": "query_patch not found"}

        query_index = filenames.index(query_patch)
        query_embedding = embeddings[query_index]

        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices[1: top_k + 1]:
            results.append({
                "patch_name": filenames[idx],
                "similarity": float(similarities[idx])
            })

        return {
            "query_patch": query_patch,
            "results": results
        }        