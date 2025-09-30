# inference.py — unified, safe inference interface
# - Exposes convert_voice(input_wav_path, output_wav_path="/tmp/output.wav", pitch_shift=0)
# - Attempts to load models if available; if not, falls back to copying input -> output.
# - Designed to be safe to import in Colab / wrapper environments.

MODULE_DIR = os.path.dirname(__file__)  # inference.py가 있는 폴더

import os
import glob
import shutil
import logging
import time

logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

# === Config: expected model locations (adjust if you use different names) ===
HUBERT_CANDIDATES = [
    os.path.join("models", "hubert", "hubert_base.pt"),
    os.path.join("models", "hubert", "hubert_soft.pt"),
    os.path.join("hubert", "hubert_base.pt"),
    os.path.join("hubert", "hubert_soft.pt"),
]
ELON_DIR = os.path.join("models", "elon")    # look for .pth/.pt/.zip inside

# module-level cache for loaded models (so repeated calls are fast)
_models = {
    "hubert": None,
    "rvc": None,
    "loaded": False,
    "hubert_path": None,
    "rvc_path": None,
}

# Helper: find hubert file
def find_hubert():
    for p in HUBERT_CANDIDATES:
        if os.path.exists(p):
            return p
    # also try scanning common locations
    alt = glob.glob(os.path.join("models", "**", "hubert*.pt"), recursive=True)
    if alt:
        return alt[0]
    return None

# Helper: find RVC/elon model candidate (.pth/.pt or zip)
def find_elon_candidate(elon_dir=ELON_DIR):
    if not os.path.isdir(elon_dir):
        return None
    candidates = glob.glob(os.path.join(elon_dir, "**", "*.pth"), recursive=True) + \
                 glob.glob(os.path.join(elon_dir, "**", "*.pt"), recursive=True) + \
                 glob.glob(os.path.join(elon_dir, "*.pth")) + glob.glob(os.path.join(elon_dir, "*.pt"))
    candidates = [c for c in candidates if "hubert" not in os.path.basename(c).lower()]
    if candidates:
        # choose largest (heuristic)
        return sorted(candidates, key=os.path.getsize, reverse=True)[0]
    zips = glob.glob(os.path.join(elon_dir, "*.zip"))
    if zips:
        return zips[0]
    return None

# Helper: if zipped model, unpack to temp dir and return candidate .pth path (or zip path unchanged)
def maybe_unzip_get_model(path):
    if not path:
        return None
    if path.endswith(".zip"):
        import tempfile, zipfile
        tmpd = tempfile.mkdtemp(prefix="rvc_model_")
        logger.info("Unzipping model zip %s -> %s", path, tmpd)
        try:
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(tmpd)
            # look for .pth/.pt inside
            found = glob.glob(os.path.join(tmpd, "**", "*.pth"), recursive=True) + \
                    glob.glob(os.path.join(tmpd, "**", "*.pt"), recursive=True)
            if found:
                return found[0]
            else:
                logger.warning("No .pth/.pt found inside zip %s", path)
                return None
        except Exception:
            logger.exception("zip extraction failed")
            return None
    else:
        return path

# === load_models: attempt to load hubert & rvc models ===
def load_models(device="cuda"):
    """
    Try to locate and (where possible) load model artifacts.
    NOTE: actual RVC/hubert python loader code depends on your implementation.
    This function attempts to load via torch when possible; if not, it will
    set placeholder entries so the pipeline can still run (fallback).
    Replace the TODO sections with your actual loader calls when ready.
    """
    import torch

    if _models["loaded"]:
        return _models

    # find hubert
    hubert_path = find_hubert()
    if hubert_path:
        logger.info("Found hubert at: %s", hubert_path)
        _models["hubert_path"] = hubert_path
        # TODO: replace with actual hubert model loader if you have one:
        try:
            # example: hubert_model = load_your_hubert(hubert_path)
            # we'll keep a placeholder
            _models["hubert"] = {"path": hubert_path}
        except Exception:
            logger.exception("Hubert load attempt failed, continuing with placeholder.")
    else:
        logger.warning("No hubert model found. Expected paths: %s", HUBERT_CANDIDATES)

    # find elon/rvc model candidate
    elon_candidate = find_elon_candidate()
    if elon_candidate:
        logger.info("Found RVC model candidate: %s", elon_candidate)
        elon_model_path = maybe_unzip_get_model(elon_candidate) or elon_candidate
        _models["rvc_path"] = elon_model_path
        try:
            # attempt torch.load for .pth/.pt if present
            if elon_model_path and elon_model_path.endswith((".pth", ".pt")):
                # DO NOT ASSUME direct torch.load is all that's needed; this is a heuristic
                try:
                    logger.info("Attempting torch.load(%s)", elon_model_path)
                    loaded = torch.load(elon_model_path, map_location=device)
                    _models["rvc"] = loaded
                    logger.info("Torch load succeeded (may still require model class wrapping).")
                except Exception:
                    logger.exception("torch.load failed; keeping path as reference.")
                    _models["rvc"] = {"path": elon_model_path}
            else:
                # zip or something else; keep path
                _models["rvc"] = {"path": elon_model_path or elon_candidate}
        except Exception:
            logger.exception("RVC model load attempt failed; using placeholders.")
    else:
        logger.warning("No RVC model candidate found under %s", ELON_DIR)

    _models["loaded"] = True
    return _models

# === main API: convert_voice ===
def convert_voice(input_wav_path, output_wav_path="/tmp/output.wav", pitch_shift=0):
    """
    High-level convert function:
      - input_wav_path: path to input wav (or any format librosa can read)
      - output_wav_path: desired path for produced wav
      - pitch_shift: optional pitch shift (semitones) — not implemented by default
    Returns path to produced wav (output_wav_path).
    Behavior:
      - tries to use your RVC/hubert models if loadable
      - if real inference cannot be executed, falls back to copying input -> output
    """
    start = time.time()
    if not os.path.exists(input_wav_path):
        raise FileNotFoundError("input missing: " + str(input_wav_path))

    # ensure models loaded (or placeholder)
    load_models(device=("cuda" if _torch_cuda_available() else "cpu"))

    # === TODO: Replace below placeholder inference with your actual inference call ===
    # Example (pseudo):
    #   result = my_rvc_infer(input_wav_path, out=output_wav_path,
    #                         hubert=_models['hubert'], model=_models['rvc'], pitch_shift=pitch_shift)
    #   return result
    #
    # For now: attempt a simple copy (so pipeline doesn't break), but keep an explicit log.
    try:
        # If you have a runner module (rvc_runner) with run_inference, try it dynamically:
        try:
            import importlib
            runner = importlib.import_module("rvc_runner")
            if hasattr(runner, "run_inference"):
                logger.info("Calling rvc_runner.run_inference(...)")
                res = runner.run_inference(input_wav_path, output_wav_path,
                                           hubert_path=_models.get("hubert_path"),
                                           model_path=_models.get("rvc_path"),
                                           pitch_shift=pitch_shift)
                # If runner returns path, use it
                if isinstance(res, str) and os.path.exists(res):
                    logger.info("rvc_runner returned file: %s", res)
                    shutil.copy2(res, output_wav_path)
                    logger.info("Copied runner output -> %s", output_wav_path)
                    return output_wav_path
                # else continue to fallback
        except Exception:
            # runner not available — that's OK
            logger.debug("No rvc_runner or it failed; will try other options or fallback.")

        # If a simple API is available in the repo (e.g., a local inference_impl module), try it:
        try:
            import importlib
            impl = importlib.import_module("inference_impl")
            if hasattr(impl, "convert_voice"):
                logger.info("Using local inference_impl.convert_voice(...)")
                res = impl.convert_voice(input_wav_path, output_wav_path=output_wav_path, pitch_shift=pitch_shift)
                if isinstance(res, str) and os.path.exists(res):
                    return res
        except Exception:
            logger.debug("No inference_impl or failed.")

        # Final fallback: copy input -> output
        logger.warning("Falling back: copying input -> output (no conversion).")
        shutil.copy2(input_wav_path, output_wav_path)
        return output_wav_path

    finally:
        elapsed = time.time() - start
        logger.info("convert_voice finished in %.2fs, output: %s", elapsed, output_wav_path)

def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# convenience warmup
def warmup():
    load_models(device=("cuda" if _torch_cuda_available() else "cpu"))
    logger.info("warmup done")
