from typing import Generic, TypeVar

T = TypeVar("T")


class DelegatingBuilder(Generic[T]):
    def __init__(self, *a, **k):
        pass

    def build(self, *a, **k):
        raise RuntimeError("multigpu disabled")
