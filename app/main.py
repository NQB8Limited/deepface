from __future__ import annotations

import base64
import binascii
import hmac
import math
import os
from typing import Any

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

DEFAULT_MODEL = os.getenv("DEEPFACE_MODEL_NAME", "ArcFace")
DEFAULT_DETECTOR = os.getenv("DEEPFACE_DETECTOR_BACKEND", "retinaface")
DEFAULT_COSINE_THRESHOLD = float(os.getenv("DEEPFACE_COSINE_THRESHOLD", "0.68"))


def verify_internal_secret(
    x_biometric_secret: str | None = Header(default=None, alias="X-Biometric-Secret"),
) -> None:
    expected = os.getenv("DEEPFACE_SHARED_SECRET", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DEEPFACE_SHARED_SECRET is not configured.",
        )

    if not x_biometric_secret or not hmac.compare_digest(x_biometric_secret, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid biometric sidecar secret.",
        )


class EmbedRequest(BaseModel):
    image_base64: str = Field(..., min_length=16)
    model_name: str = Field(default=DEFAULT_MODEL)
    detector_backend: str = Field(default=DEFAULT_DETECTOR)


class EmbedResponse(BaseModel):
    is_real: bool
    anti_spoof_score: float | None = None
    embedding: list[float] | None = None
    model_name: str
    detector_backend: str
    reason: str | None = None


class VerifyRequest(BaseModel):
    image_base64: str = Field(..., min_length=16)
    reference_embedding: list[float] = Field(..., min_length=64)
    model_name: str = Field(default=DEFAULT_MODEL)
    detector_backend: str = Field(default=DEFAULT_DETECTOR)
    threshold: float = Field(default=DEFAULT_COSINE_THRESHOLD, gt=0.0, lt=2.0)


class VerifyResponse(BaseModel):
    is_real: bool
    verified: bool
    anti_spoof_score: float | None = None
    distance: float | None = None
    threshold: float
    model_name: str
    detector_backend: str
    reason: str | None = None


def decode_image_in_memory(payload: str) -> np.ndarray[Any, Any]:
    cleaned = payload.split(",", 1)[1] if "," in payload and payload.startswith("data:") else payload
    try:
        raw_bytes = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid base64 image payload.",
        ) from exc

    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    del raw_bytes, buffer

    if image is None or image.size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to decode image bytes.",
        )

    return image


def cosine_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Embedding dimensions do not match.",
        )

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0

    similarity = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return 1.0 - similarity


def _run_liveness_and_embed(
    image: np.ndarray[Any, Any],
    model_name: str,
    detector_backend: str,
) -> tuple[bool, float | None, list[float] | None, str | None]:
    # Non-inference guardrail (REQ-038 / BR-028): never use demographic/emotion attribute inference.
    from deepface import DeepFace

    try:
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend=detector_backend,
            enforce_detection=True,
            anti_spoofing=True,
        )
    except ValueError:
        return False, None, None, "NO_SINGLE_FACE_DETECTED"

    if len(faces) != 1:
        return False, None, None, "MULTIPLE_FACES_DETECTED"

    face_meta = faces[0]
    is_real = bool(face_meta.get("is_real", False))
    raw_score = face_meta.get("antispoof_score")
    anti_spoof_score = float(raw_score) if raw_score is not None else None

    if not is_real:
        return False, anti_spoof_score, None, "LIVENESS_FAILED"

    representations = DeepFace.represent(
        img_path=image,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=True,
    )
    if len(representations) != 1 or "embedding" not in representations[0]:
        return False, anti_spoof_score, None, "EMBEDDING_EXTRACTION_FAILED"

    vector = [float(v) for v in representations[0]["embedding"]]
    return True, anti_spoof_score, vector, None


app = FastAPI(title="Presence Biometric Sidecar (DeepFace)", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "engine": "deepface",
        "model": DEFAULT_MODEL,
        "detector": DEFAULT_DETECTOR,
    }


@app.post(
    "/v1/liveness-and-embed",
    response_model=EmbedResponse,
    dependencies=[Depends(verify_internal_secret)],
)
def liveness_and_embed(request: EmbedRequest) -> EmbedResponse:
    image = decode_image_in_memory(request.image_base64)
    try:
        is_real, score, vector, reason = _run_liveness_and_embed(
            image=image,
            model_name=request.model_name,
            detector_backend=request.detector_backend,
        )
        return EmbedResponse(
            is_real=is_real,
            anti_spoof_score=score,
            embedding=vector,
            model_name=request.model_name,
            detector_backend=request.detector_backend,
            reason=reason,
        )
    finally:
        del image


@app.post(
    "/v1/liveness-and-verify",
    response_model=VerifyResponse,
    dependencies=[Depends(verify_internal_secret)],
)
def liveness_and_verify(request: VerifyRequest) -> VerifyResponse:
    image = decode_image_in_memory(request.image_base64)
    try:
        is_real, score, vector, reason = _run_liveness_and_embed(
            image=image,
            model_name=request.model_name,
            detector_backend=request.detector_backend,
        )
        if not is_real or vector is None:
            return VerifyResponse(
                is_real=False,
                verified=False,
                anti_spoof_score=score,
                distance=None,
                threshold=request.threshold,
                model_name=request.model_name,
                detector_backend=request.detector_backend,
                reason=reason or "LIVENESS_FAILED",
            )

        distance = cosine_distance(vector, request.reference_embedding)
        verified = distance <= request.threshold

        return VerifyResponse(
            is_real=True,
            verified=verified,
            anti_spoof_score=score,
            distance=distance,
            threshold=request.threshold,
            model_name=request.model_name,
            detector_backend=request.detector_backend,
            reason=None if verified else "FACE_MISMATCH",
        )
    finally:
        del image
