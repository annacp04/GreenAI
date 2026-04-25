import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class AirQualityAnalyzer:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """Initialize the CLIP model and processor."""
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def analyze_image(self, image_path):
        """
        Performs dual analysis: 
        1. Zero-shot classification probabilities.
        2. Direct cosine similarity for a pollution intensity score.
        """
        image = Image.open(image_path).convert("RGB")

        # Define labels for classification and intensity poles
        classification_labels = [
            "a clear blue sky", 
            "a very foggy and polluted city", 
            "heavy smog and low visibility", 
            "cloudy and humid weather",
            "a photo with high air particulate matter"
        ]
        
        intensity_prompts = [
            "high air pollution, thick brown smog, low visibility, dirty atmosphere", 
            "perfectly clear blue sky, high visibility, clean fresh air, no pollution"
        ]

        # Combine all text for a single pass through the model
        all_texts = classification_labels + intensity_prompts
        
        inputs = self.processor(
            text=all_texts, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 1. Get Classification Probabilities
            # Focus only on the first few labels for softmax
            logits_per_image = outputs.logits_per_image[:, :len(classification_labels)]
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            # 2. Get Cosine Similarity Scores
            # Normalize embeddings for manual cosine similarity
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            
            # Slice only the intensity prompts (the last two)
            pollution_text_embeds = text_embeds[len(classification_labels):]
            similarities = torch.matmul(image_embeds, pollution_text_embeds.t()).cpu().numpy()[0]

        return {
            "probs": dict(zip(classification_labels, probs)),
            "polluted_score": similarities[0],
            "clean_score": similarities[1],
            "balance": similarities[0] - similarities[1],
            "image_vector": outputs.image_embeds.cpu().numpy()
        }