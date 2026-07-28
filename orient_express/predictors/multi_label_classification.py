from dataclasses import dataclass

from PIL import Image

from .predictor import ImagePredictor


@dataclass
class MultiLabelClassificationPrediction:
    classes: list[str]
    class_scores: dict[str, float]

    def to_dict(self):
        return {
            "classes": self.classes,
            "class_scores": self.class_scores,
        }


class MultiLabelClassificationPredictor(ImagePredictor):
    model_type = "multi-label-classification-onnx"

    def predict(
        self, images: list[Image.Image], confidence: float
    ) -> list[MultiLabelClassificationPrediction]:
        if not images:
            return []
        feed = self.preprocess(images)
        return self.postprocess(self.infer(feed), feed, confidence=confidence)

    def postprocess(
        self, outputs, feed, confidence: float
    ) -> list[MultiLabelClassificationPrediction]:
        results = []
        for class_scores in outputs[0]:
            classes = []
            # self.classes is 1-indexed
            for idx, score in enumerate(class_scores):
                if score > confidence:
                    classes.append(self.classes.get(idx + 1, "Unknown"))
            results.append(
                MultiLabelClassificationPrediction(
                    classes=classes,
                    class_scores={
                        # self.classes is 1-indexed
                        clss: float(class_scores[class_idx - 1])
                        for class_idx, clss in self.classes.items()
                    },
                )
            )
        return results

    def get_annotated_image(
        self, image: Image.Image, predictions: MultiLabelClassificationPrediction
    ):
        return None
