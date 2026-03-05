class EmbeddingModel:

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        if "UNI2" in self.model_name:
            self._load_uni2()
        elif "Virchow2" in self.model_name:
            self._load_virchow2()
        elif "path-foundation" in self.model_name:
            self._load_path_foundation()
        else:
            raise ValueError(f"Unsupported model {self.model_name}")

    def _load_uni2(self):
        from transformers import AutoModel, AutoImageProcessor
        import torch

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_virchow2(self):
        from transformers import AutoModel, AutoImageProcessor
        import torch

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _load_path_foundation(self):
        from huggingface_hub import snapshot_download
        import tensorflow as tf

        model_path = snapshot_download(self.model_name)
        self.model = tf.keras.layers.TFSMLayer(
            model_path,
            call_endpoint="serving_default"
    )

    def embed(self, image):
        if self.model == 'path-foundation':
            return self.embed_tf(image)
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

    def embed_tf(self, image):
        tensor = tf.cast(tf.expand_dims(np.array(image), 0), tf.float32) / 255.0
        output = self.model.infer(tensor)
        return output["output_0"].numpy().flatten()