import os
import json
from typing import Dict, List, Optional, Tuple
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CLIPSearch:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model for image-text search
        
        Args:
            model_name: Hugging Face model name for CLIP
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.load_model()
    
    def load_model(self):
        """Load CLIP model and processor"""
        try:
            print(f"Loading CLIP model: {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move to device
            self.model.to(self.device)
            print(f"CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            self.model = None
            self.processor = None
    
    def get_image_features(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Extract features from an image using CLIP
        
        Args:
            image_path: Path to input image
            
        Returns:
            Image features tensor or None if error
        """
        if self.model is None or self.processor is None:
            return None
            
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features
            
        except Exception as e:
            print(f"Error extracting image features: {e}")
            return None
    
    def get_text_features(self, text: str) -> Optional[torch.Tensor]:
        """
        Extract features from text using CLIP
        
        Args:
            text: Input text
            
        Returns:
            Text features tensor or None if error
        """
        if self.model is None or self.processor is None:
            return None
            
        try:
            # Preprocess text
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get text features
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features
            
        except Exception as e:
            print(f"Error extracting text features: {e}")
            return None
    
    def get_image_overview(self, image_path: str, 
                          descriptive_phrases: List[str] = None) -> Dict:
        """
        Generate natural language description of an image
        
        Args:
            image_path: Path to input image
            descriptive_phrases: List of phrases to test against image
            
        Returns:
            Dictionary with image description and confidence scores
        """
        if self.model is None or self.processor is None:
            return {"error": "CLIP model not loaded", "description": "Unable to analyze image"}
        
        # Default descriptive phrases if none provided
        if descriptive_phrases is None:
            descriptive_phrases = [
                "a person using a laptop",
                "a person holding a phone",
                "a person with a bottle",
                "a person with a cow",
                "a person working on computer",
                "a person taking a selfie",
                "a person drinking water",
                "a person with an animal",
                "a person in an office",
                "a person at home",
                "a person outdoors",
                "a person indoors",
                "a person smiling",
                "a person looking at camera",
                "a person not looking at camera"
            ]
        
        try:
            # Get image features
            image_features = self.get_image_features(image_path)
            if image_features is None:
                return {"error": "Could not extract image features"}
            
            # Get text features for all phrases
            text_features_list = []
            valid_phrases = []
            
            for phrase in descriptive_phrases:
                text_features = self.get_text_features(phrase)
                if text_features is not None:
                    text_features_list.append(text_features)
                    valid_phrases.append(phrase)
            
            if not text_features_list:
                return {"error": "Could not extract text features"}
            
            # Calculate similarities
            similarities = []
            for text_features in text_features_list:
                similarity = torch.cosine_similarity(image_features, text_features, dim=1)
                similarities.append(similarity.item())
            
            # Find best matches
            phrase_scores = list(zip(valid_phrases, similarities))
            phrase_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Generate description
            top_phrases = phrase_scores[:3]
            description_parts = []
            
            for phrase, score in top_phrases:
                if score > 0.2:  # Threshold for relevance
                    description_parts.append(f"{phrase} ({score:.2f})")
            
            description = " | ".join(description_parts) if description_parts else "No clear description available"
            
            return {
                "description": description,
                "top_matches": phrase_scores[:5],
                "image_path": image_path,
                "analysis_phrases": len(valid_phrases)
            }
            
        except Exception as e:
            return {"error": str(e), "description": "Error analyzing image"}
    
    def search_with_text(self, text_query: str, image_paths: List[str]) -> Dict:
        """
        Search images using text query
        
        Args:
            text_query: Text query to search for
            image_paths: List of image paths to search through
            
        Returns:
            Dictionary with similarity scores and ranked results
        """
        if self.model is None or self.processor is None:
            return {"error": "CLIP model not loaded", "results": []}
        
        try:
            # Get text features
            text_features = self.get_text_features(text_query)
            if text_features is None:
                return {"error": "Could not extract text features", "results": []}
            
            # Get image features for all images
            image_features_list = []
            valid_images = []
            
            for image_path in image_paths:
                if os.path.exists(image_path):
                    image_features = self.get_image_features(image_path)
                    if image_features is not None:
                        image_features_list.append(image_features)
                        valid_images.append(image_path)
            
            if not image_features_list:
                return {"error": "No valid images found", "results": []}
            
            # Calculate similarities
            similarities = []
            for image_features in image_features_list:
                similarity = torch.cosine_similarity(text_features, image_features, dim=1)
                similarities.append(similarity.item())
            
            # Rank results
            results = list(zip(valid_images, similarities))
            results.sort(key=lambda x: x[1], reverse=True)
            
            return {
                "query": text_query,
                "results": [
                    {
                        "image_path": path,
                        "similarity_score": score,
                        "rank": i + 1
                    }
                    for i, (path, score) in enumerate(results)
                ],
                "total_images": len(results)
            }
            
        except Exception as e:
            return {"error": str(e), "results": []}
    
    def compare_images(self, image_paths: List[str]) -> Dict:
        """
        Compare multiple images and find similarities
        
        Args:
            image_paths: List of image paths to compare
            
        Returns:
            Dictionary with similarity matrix and analysis
        """
        if self.model is None or self.processor is None:
            return {"error": "CLIP model not loaded", "similarities": []}
        
        try:
            # Get features for all images
            image_features_list = []
            valid_images = []
            
            for image_path in image_paths:
                if os.path.exists(image_path):
                    image_features = self.get_image_features(image_path)
                    if image_features is not None:
                        image_features_list.append(image_features.cpu().numpy())
                        valid_images.append(image_path)
            
            if len(image_features_list) < 2:
                return {"error": "Need at least 2 valid images", "similarities": []}
            
            # Calculate similarity matrix
            features_matrix = np.vstack(image_features_list)
            similarity_matrix = cosine_similarity(features_matrix)
            
            # Find most similar pairs
            similarities = []
            for i in range(len(valid_images)):
                for j in range(i + 1, len(valid_images)):
                    similarities.append({
                        "image1": valid_images[i],
                        "image2": valid_images[j],
                        "similarity": float(similarity_matrix[i][j])
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "similarity_matrix": similarity_matrix.tolist(),
                "image_paths": valid_images,
                "most_similar_pairs": similarities[:5],
                "total_comparisons": len(similarities)
            }
            
        except Exception as e:
            return {"error": str(e), "similarities": []}

# Convenience functions for easy import
def get_image_overview(image_path: str, descriptive_phrases: List[str] = None) -> Dict:
    """Generate natural language description of an image"""
    clip_search = CLIPSearch()
    return clip_search.get_image_overview(image_path, descriptive_phrases)

def search_with_text(text_query: str, image_paths: List[str]) -> Dict:
    """Search images using text query"""
    clip_search = CLIPSearch()
    return clip_search.search_with_text(text_query, image_paths)

def compare_images(image_paths: List[str]) -> Dict:
    """Compare multiple images and find similarities"""
    clip_search = CLIPSearch()
    return clip_search.compare_images(image_paths)

if __name__ == "__main__":
    # Test CLIP functionality
    clip_search = CLIPSearch()
    
    # Test with a sample image if available
    test_image = "../data/test/laptop/laptop.png"
    if os.path.exists(test_image):
        print("Testing image overview...")
        overview = clip_search.get_image_overview(test_image)
        print("Overview Results:")
        print(json.dumps(overview, indent=2))
        
        print("\nTesting text search...")
        search_results = clip_search.search_with_text("laptop computer", [test_image])
        print("Search Results:")
        print(json.dumps(search_results, indent=2))
    else:
        print("No test image found. Please add an image to test CLIP functionality.")
