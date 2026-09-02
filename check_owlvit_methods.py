#!/usr/bin/env python
"""Check which post_process methods are actually available in OWL-ViT"""
import sys

try:
    from transformers import OwlViTProcessor
    import transformers
    
    print(f"\n=== Transformers Version: {transformers.__version__} ===\n")
    
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    
    print("=== Available post_process methods on processor ===")
    methods = [m for m in dir(processor) if "post_process" in m.lower() and not m.startswith('_')]
    for method in methods:
        print(f"  - processor.{method}")
    
    print("\n=== Available post_process methods on image_processor ===")
    if hasattr(processor, 'image_processor'):
        img_methods = [m for m in dir(processor.image_processor) if "post_process" in m.lower() and not m.startswith('_')]
        for method in img_methods:
            print(f"  - processor.image_processor.{method}")
    
    print("\n=== Available post_process methods on feature_extractor ===")
    if hasattr(processor, 'feature_extractor'):
        feat_methods = [m for m in dir(processor.feature_extractor) if "post_process" in m.lower() and not m.startswith('_')]
        for method in feat_methods:
            print(f"  - processor.feature_extractor.{method}")
    
    print("\n=== Recommended Usage ===")
    if methods:
        print(f"  Use: processor.{methods[0]}()")
    elif hasattr(processor, 'image_processor') and img_methods:
        print(f"  Use: processor.image_processor.{img_methods[0]}()")
    else:
        print("  Manual post-processing required - no built-in method found")
        
except ImportError as e:
    print(f"ERROR: transformers not installed or OWL-ViT not available: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
