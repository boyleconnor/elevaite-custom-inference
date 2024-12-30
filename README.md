# ElevAIte Custom Inference Code

This repo is a code template to allow developers to quickly & easily add previously-unsupported model architectures to
[ElevAIte](https://www.iopex.com/generative-ai-solutions/).

Using this template requires basic knowledge of Python, Git, and GitHub, as well as how to load and run inference on the
model you wish to support.

## Developing new Inference Code

1) Create a copy of this template to your own GitHub account (`"Use this template" -> "Create a new repository"`)
2) Clone your copy of this repository
3) Install requirements in [`requirements.txt`](requirements.txt) to your Python environment (e.g. by running
   `pip install -r requirements.txt`; you may want to set up
   a [virtual environment](https://docs.python.org/3/library/venv.html))
4) (Optional) [Install pre-commit](https://pre-commit.com/#introduction) for code checks on every Git commit
5) Edit `CustomDeployment.__init__()` in [`inference.py`](inference.py) to implement loading the model into memory (e.g.
   RAM or VRAM, depending on the value of `device`)
6) Edit `CustomDeployment.infer()` in [`inference.py`](inference.py) to implement running inference on the model.
7) Add other paths as desired
8) Test by running: `serve run inference:app_builder device=cpu model_path=<PATH_TO_MODEL>`, replacing `<PATH_TO_MODEL>`
   with the path to the model that you wish to load. (Alternatively, you can use `device=cuda` instead of `device=cpu`
   if running on a GPU-enabled machine with the appropriate drivers and a GPU version of PyTorch; this will also require
   changing the `deployment` to `@serve.deployment(ray_actor_options={"num_gpus": 1})`, or whatever number of GPUs you
   have available, if not `1`.)
9) After the model has finished loading, navigate to [`http://localhost:8000/docs`](http://localhost:8000/docs) to
   interact with the model via a Swagger web UI.
