"""CLIP model loading. torch and open_clip are imported lazily so the app starts (and tests run) without them."""
from PIL import Image


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def no_grad():
    import torch

    return torch.no_grad()


def load_clip_model(model_name, pretrained, device):
    """Load CLIP model, return (model, preprocess, tokenizer, dimension)."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()

    dummy_image = Image.new('RGB', (224, 224))
    dummy_input = preprocess(dummy_image).unsqueeze(0).to(device)
    with no_grad():
        dummy_features = model.encode_image(dummy_input)
        dimension = dummy_features.shape[-1]

    return model, preprocess, tokenizer, dimension
