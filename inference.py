import logging

from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve
from ray.serve import Application

logger = logging.getLogger("ray.serve")


class InferenceRequest(BaseModel):
    # TODO: Add fields; this model defines the format that callers will use
    #       to make inference requests
    pass


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
        # # Load the model into memory:
        # self.model = load_model(model_path, device)  # TODO: Implement
        raise NotImplementedError()

    @web_app.post("/infer")
    def infer(self, inference_request: InferenceRequest) -> dict:
        """
        TODO: This docstring will be displayed on the OpenAPI spec &
              auto-generated Swagger web UI. Fill it with useful information
              about running & using this model deployment.

        :param inference_request:
        :return:
        """
        # input_one = inference_request.some_field
        # input_two = inference_request.some_other_field
        raise NotImplementedError(
            "This model's inference logic has not been implemented!"
        )

    # TODO: Add other routes with other functionality, if desired
    @web_app.get("/other_route")
    def other_functionality(self):
        raise NotImplementedError()


def app_builder(args: dict) -> Application:
    return CustomDeployment.bind(  # type: ignore[attr-defined]
        args["model_path"],
        args["device"],
    )
