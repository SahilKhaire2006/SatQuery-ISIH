#!/usr/bin/env python
"""Check the exact signature of post_process_grounded_object_detection"""
from transformers import OwlViTProcessor
import inspect

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

print("\n=== Signature of post_process_grounded_object_detection ===\n")
method = processor.post_process_grounded_object_detection
print(inspect.signature(method))
print("\n=== Help ===")
help(method)
