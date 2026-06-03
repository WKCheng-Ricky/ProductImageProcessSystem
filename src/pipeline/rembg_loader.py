try:
    import rembg
    _REMBG_AVAILABLE = True
except ImportError:
    _REMBG_AVAILABLE = False


def load_rembg_session(model_name, providers_list):
    """Load and return a rembg session."""
    if not _REMBG_AVAILABLE:
        raise ImportError("Please install rembg: pip install rembg")
    return rembg.new_session(model_name, providers=providers_list)
