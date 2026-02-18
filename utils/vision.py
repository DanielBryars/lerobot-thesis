"""Object detection utilities using GroundingDINO for scene understanding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class Detection:
    """A single detected object."""

    label: str
    center: tuple[float, float]  # normalized (x, y) in [0, 1]
    box: tuple[float, float, float, float]  # normalized (x1, y1, x2, y2)
    confidence: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "center": {"x": self.center[0], "y": self.center[1]},
            "box": {
                "x1": self.box[0],
                "y1": self.box[1],
                "x2": self.box[2],
                "y2": self.box[3],
            },
            "confidence": round(self.confidence, 4),
        }


class ObjectDetector:
    """Zero-shot object detector using GroundingDINO.

    Uses IDEA-Research/grounding-dino-tiny from HuggingFace transformers.
    Model is loaded lazily on first detect() call.

    Example:
        detector = ObjectDetector()
        detections = detector.detect(image, ["white block", "bowl"])
    """

    MODEL_ID = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self._model_id = model_id or self.MODEL_ID
        self._device = device
        self._processor = None
        self._model = None

    def _load_model(self):
        """Load model and processor on first use."""
        if self._model is not None:
            return

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        print(f"Loading {self._model_id} on {device}...")
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._model_id
        ).to(device)
        self._model.eval()
        print("Model loaded.")

    def detect(
        self,
        image: Image.Image,
        labels: list[str],
        threshold: float = 0.3,
    ) -> list[Detection]:
        """Detect objects in an image.

        Args:
            image: PIL Image to analyze.
            labels: List of text labels to detect (e.g. ["white block", "bowl"]).
            threshold: Minimum confidence threshold for detections.

        Returns:
            List of Detection objects, sorted by confidence (highest first).
        """
        self._load_model()

        # GroundingDINO expects labels as a single period-separated string
        text = ". ".join(labels) + "."

        inputs = self._processor(images=image, text=text, return_tensors="pt").to(
            self._device
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        w, h = image.size
        detections = []

        # Use text_labels if available (transformers >= 4.51), fall back to labels
        label_key = "text_labels" if "text_labels" in results else "labels"
        for box, score, label in zip(
            results["boxes"], results["scores"], results[label_key]
        ):
            x1, y1, x2, y2 = box.tolist()
            # Normalize to [0, 1]
            nx1, ny1 = x1 / w, y1 / h
            nx2, ny2 = x2 / w, y2 / h
            cx = (nx1 + nx2) / 2
            cy = (ny1 + ny2) / 2

            detections.append(
                Detection(
                    label=label.strip(),
                    center=(round(cx, 4), round(cy, 4)),
                    box=(round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)),
                    confidence=round(score.item(), 4),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
