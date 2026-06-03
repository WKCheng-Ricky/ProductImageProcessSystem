import torch
import open_clip
from PIL import Image


def load_clip_model(model_name, pretrained, device):
    """Load CLIP model, return (model, preprocess, tokenizer, dimension)."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()

    dummy_image = Image.new('RGB', (224, 224))
    dummy_input = preprocess(dummy_image).unsqueeze(0).to(device)
    with torch.no_grad():
        dummy_features = model.encode_image(dummy_input)
        dimension = dummy_features.shape[-1]

    return model, preprocess, tokenizer, dimension
