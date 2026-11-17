"""State-aware motion generation with the pinned primary pipeline."""


def __getattr__(name):
    if name in {"MotionPipeline", "MotionResult"}:
        from inference.pipeline import MotionPipeline, MotionResult
        return {"MotionPipeline": MotionPipeline, "MotionResult": MotionResult}[name]
    raise AttributeError(name)


__all__ = ["MotionPipeline", "MotionResult"]
