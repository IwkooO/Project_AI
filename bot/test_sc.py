from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

url = "https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

choices = ["San Jose", "San Diego", "Los Angeles", "Las Vegas", "San Francisco"]
inputs = processor(text=choices, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print(probs)