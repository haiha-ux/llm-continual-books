"""Fetch the pinned base model into the HF cache (resumes partial files).  python tools/download_base.py"""
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28"))
