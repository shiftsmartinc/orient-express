from dataclasses import dataclass

from PIL import Image

from .predictor import ImagePredictor


@dataclass
class ClassificationPrediction:
    clss: str
    score: float
    class_scores: dict[str, float]

    def to_dict(self):
        return {
            "class": self.clss,
            "score": self.score,
            "class_scores": self.class_scores,
        }


class ClassificationPredictor(ImagePredictor):
    model_type = "classification-onnx"

    def predict(self, images: list[Image.Image]) -> list[ClassificationPrediction]:
        if not images:
            return []
        feed = self.preprocess(images)
        return self.postprocess(self.infer(feed), feed)

    def postprocess(self, outputs, feed) -> list[ClassificationPrediction]:
        results = []
        for class_scores in outputs[0]:
            max_class_idx = class_scores.argmax()
            # self.classes is 1-indexed
            max_clss = self.classes.get(max_class_idx + 1, "Unknown")
            results.append(
                ClassificationPrediction(
                    clss=max_clss,
                    score=float(class_scores[max_class_idx]),
                    class_scores={
                        # self.classes is 1-indexed
                        clss: float(class_scores[class_idx - 1])
                        for class_idx, clss in self.classes.items()
                    },
                )
            )
        return results

    def get_annotated_image(
        self, image: Image.Image, predictions: ClassificationPrediction
    ):
        return None

    def to_response(self, image, prediction, include_debug: bool = True):
        # Historical (pre-to_response) response shape for classification:
        # flat prediction fields with a status key, no debug image.
        return {**prediction.to_dict(), "status": "success"}
