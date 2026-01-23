# api/settings.py
import os

API_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(API_DIR)

CONFIGS_ROOT = os.path.join(PROJECT_ROOT, "configs")

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "models")