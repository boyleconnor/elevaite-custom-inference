import logging

import numpy as np
import torch
from PIL import Image
from colpali_engine import ColPali, ColPaliProcessor
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ray import serve
from ray.serve import Application

logger = logging.getLogger("ray.serve")


class InferenceRequest(BaseModel):
    queries: list[str]


web_app = FastAPI()


@serve.deployment
@serve.ingress(web_app)
# NOTE: The name of this class must match the name of this application's
#       deployment on ElevAIte's RayService; i.e. don't rename it unless
#       you are sure you know what you're doing.
class CustomDeployment:
    def __init__(
        self,
        model_path: str,
        device: str,
    ):
        # Load the model into memory:
        self.model: ColPali = ColPali.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        ).eval()
        print(f"Type of model: {type(self.model)}")
        self.processor: ColPaliProcessor = ColPaliProcessor.from_pretrained(model_path)
        print(f"Type of processor: {type(self.processor)}")

    @web_app.post("/infer")
    def infer_query(self, inference_request: InferenceRequest) -> list:
        """
        TODO: This docstring will be displayed on the OpenAPI spec &
              auto-generated Swagger web UI. Fill it with useful information
              about running & using this model deployment.

        :param inference_request:
        :return:
        """
        queries = inference_request.queries

        # Process the inputs
        batch_queries = self.processor.process_queries(queries).to(self.model.device)

        # Forward pass
        with torch.no_grad():
            query_embeddings_tensor = self.model(**batch_queries)
        query_embeddings: np.ndarray = query_embeddings_tensor.cpu().float().numpy()

        # NOTE: `query_embeddings` is a NumPy array, which is not JSON
        #       serializable. Here we have to convert to a list so that
        #       FastAPI can convert it JSON.
        return query_embeddings.tolist()

    # TODO: Add other routes with other functionality, if desired
    @web_app.post("/infer_image")
    def infer_image(
        self,
        image_files: list[UploadFile] = File([])
    ):
        # Process the inputs
        images = [
            Image.open(image_file.file).convert("RGB") for image_file in image_files
        ]
        batch_images = self.processor.process_images(images).to(self.model.device)

        # Forward pass
        with torch.no_grad():
            image_embeddings_tensor = self.model(**batch_images)
        image_embeddings: np.ndarray = image_embeddings_tensor.cpu().float().numpy()

        # NOTE: `query_embeddings` is a NumPy array, which is not JSON
        #       serializable. Here we have to convert to a list so that
        #       FastAPI can convert it JSON.
        return image_embeddings.tolist()


def app_builder(args: dict) -> Application:
    return CustomDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["device"],
    )
