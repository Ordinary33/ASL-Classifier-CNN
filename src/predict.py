import torch
import cv2
from PIL import Image
from pathlib import Path
from src.model import ASLCNNModel
from src.transforms import get_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "asl_cnn_model.pth"


class ASLClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "del",
            "nothing",
            "space",
        ]
        self.model = ASLCNNModel(num_classes=len(self.classes)).to(self.device)
        self.model_path = model_path if model_path else MODELS_DIR
        self._load_model()
        self.model.eval()
        self.transform = get_transform()["val"]

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def classify_frame(self, frame):
        """
        Takes a raw OpenCV frame (BGR numpy array), converts it to RGB, transforms it, and predicts.
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
            predicted_class = self.classes[predicted_idx.item()]
            confidence_score = confidence.item()

        return predicted_class, confidence_score


if __name__ == "__main__":
    model = ASLClassifier()
    test_image_path = (
        PROJECT_ROOT / "data" / "asl_alphabet_test" / "asl_alphabet_test" / "A_test.jpg"
    )
    image = cv2.imread(str(test_image_path))
    label, confidence = model.classify_frame(image)
    print(f"Predicted Label: {label}, Confidence: {confidence:.4f}")
