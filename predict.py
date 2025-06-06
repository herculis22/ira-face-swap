import torch
from cog import BasePredictor, Input, Path
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

class Predictor(BasePredictor):
    def setup(self):
        # Load realistic image generator (Stable Diffusion)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1",
            torch_dtype=torch.float16,
            use_auth_token=True
        ).to("cuda")

        # Load face swap model
        self.face_analyzer = FaceAnalysis(name="buffalo_l")
        self.face_analyzer.prepare(ctx_id=0)
        self.face_swapper = model_zoo.get_model('inswapper_128.onnx', download=True)

    def predict(
        self,
        prompt: str = Input(description="Prompt for sexy dress woman"),
        face_image: Path = Input(description="Photo of real face")
    ) -> Path:
        # Step 1: Generate image
        image = self.pipe(prompt).images[0]

        # Step 2: Swap face
        background = np.array(image)
        face = cv2.imread(str(face_image))
        faces = self.face_analyzer.get(face)
        if not faces:
            raise Exception("No face found in uploaded image.")
        swapped = self.face_swapper.get(background, faces[0])

        # Save and return
        out_path = "/tmp/out.png"
        Image.fromarray(swapped).save(out_path)
        return Path(out_path)
