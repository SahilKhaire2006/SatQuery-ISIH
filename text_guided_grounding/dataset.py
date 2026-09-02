"""
Dataset Loader for Remote Sensing Visual Grounding (RSVG / DIOR-RSVG / OPT-RSVG)

Reads native dataset annotations containing satellite images, referring expressions (text queries),
and ground-truth bounding boxes [xmin, ymin, xmax, ymax].
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image


class RSVGDataset:
    """
    Loader for RSVG (Remote Sensing Visual Grounding) datasets such as DIOR-RSVG and OPT-RSVG.
    Supports reading native JSON/XML annotations with ground truth bounding boxes.
    """

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        annotation_file: Optional[Union[str, Path]] = None,
        split: str = "test",
        image_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize RSVG Dataset loader.

        Args:
            dataset_dir: Directory containing images and annotations
            annotation_file: Path to native annotation file (JSON or XML)
            split: Dataset split ('train', 'val', 'test')
            image_size: Optional target size tuple (width, height) to resize images
        """
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.split = split
        self.image_size = image_size
        self.samples: List[Dict[str, Any]] = []

        if self.annotation_file and self.annotation_file.exists():
            self._load_annotations()

    def _load_annotations(self):
        """Parse native annotation file (JSON or XML format)."""
        suffix = self.annotation_file.suffix.lower()

        if suffix == ".json":
            with open(self.annotation_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Support standard RSVG JSON schemas
            if isinstance(data, list):
                raw_samples = data
            elif isinstance(data, dict):
                raw_samples = data.get("annotations", data.get("images", [data]))
            else:
                raw_samples = []

            for item in raw_samples:
                sample = self._parse_json_item(item)
                if sample:
                    self.samples.append(sample)

        elif suffix == ".xml":
            # Parse Pascal VOC style XML with referring expression extensions
            tree = ET.parse(self.annotation_file)
            root = tree.getroot()
            sample = self._parse_xml_root(root)
            if sample:
                self.samples.append(sample)

    def _parse_json_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract image path, query, and bbox from JSON record."""
        img_name = item.get("file_name") or item.get("image_id") or item.get("img_path")
        query = item.get("query") or item.get("expression") or item.get("text") or item.get("caption")
        bbox = item.get("bbox") or item.get("gt_bbox") or item.get("box")

        if not img_name or not query or bbox is None:
            return None

        # Standardize bbox to [xmin, ymin, xmax, ymax]
        if len(bbox) == 4:
            # Check if format is [x, y, w, h] vs [xmin, ymin, xmax, ymax]
            if item.get("bbox_format") == "xywh" or (bbox[2] < bbox[0] or bbox[3] < bbox[1]):
                xmin, ymin, w, h = bbox
                xmax, ymax = xmin + w, ymin + h
            else:
                xmin, ymin, xmax, ymax = bbox
        else:
            return None

        img_path = self.dataset_dir / img_name if self.dataset_dir else Path(img_name)

        return {
            "image_id": str(item.get("id", img_name)),
            "image_path": str(img_path),
            "query": str(query),
            "gt_bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
            "category": item.get("category", "object")
        }

    def _parse_xml_root(self, root: ET.Element) -> Optional[Dict[str, Any]]:
        """Extract sample data from XML node."""
        filename = root.findtext("filename")
        query = root.findtext("expression") or root.findtext("query")
        bndbox = root.find(".//bndbox")

        if not filename or not query or bndbox is None:
            return None

        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        img_path = self.dataset_dir / filename if self.dataset_dir else Path(filename)

        return {
            "image_id": filename,
            "image_path": str(img_path),
            "query": query,
            "gt_bbox": [xmin, ymin, xmax, ymax],
            "category": root.findtext(".//name", "object")
        }

    def add_sample(
        self,
        image_path: Union[str, Path],
        query: str,
        gt_bbox: List[float],
        image_id: Optional[str] = None,
        category: str = "object"
    ):
        """Manually register a verified sample into dataset index."""
        self.samples.append({
            "image_id": image_id or Path(image_path).name,
            "image_path": str(image_path),
            "query": query,
            "gt_bbox": [float(b) for b in gt_bbox],
            "category": category
        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch sample by index, loading image into memory as numpy array."""
        sample = self.samples[idx]
        image_path = Path(sample["image_path"])

        if image_path.exists():
            pil_img = Image.open(image_path).convert("RGB")
            if self.image_size:
                w_orig, h_orig = pil_img.size
                pil_img = pil_img.resize(self.image_size, Image.BILINEAR)
                # Rescale GT bbox accordingly
                w_new, h_new = self.image_size
                scale_x = w_new / max(1, w_orig)
                scale_y = h_new / max(1, h_orig)
                gt = sample["gt_bbox"]
                sample_gt = [gt[0] * scale_x, gt[1] * scale_y, gt[2] * scale_x, gt[3] * scale_y]
            else:
                sample_gt = list(sample["gt_bbox"])
            img_arr = np.array(pil_img)
        else:
            # Fallback placeholder frame if file path is unreachable
            img_arr = np.zeros((512, 512, 3), dtype=np.uint8)
            sample_gt = list(sample["gt_bbox"])

        return {
            "image": img_arr,
            "query": sample["query"],
            "gt_bbox": sample_gt,
            "image_id": sample["image_id"],
            "image_path": sample["image_path"],
            "category": sample.get("category", "object")
        }
