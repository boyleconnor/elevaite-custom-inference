import logging

import numpy as np
import torch
from colpali_engine import ColPali, ColPaliProcessor
from fastapi import FastAPI
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
        self.model = ColPali.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_path)

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
    @web_app.get("/other_route")
    def other_functionality(self):
        raise NotImplementedError()


def app_builder(args: dict) -> Application:
    return CustomDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["device"],
    )
