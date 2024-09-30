from fastapi import APIRouter
from .Emotion_route import router as emotion_cls_route

router = APIRouter()
router.include_router(emotion_cls_route, prefix = "/emotion_classification")