from .code_generator import visit_withAsyncTask, visit_withAsyncTasks


def _build_dispatch():
    import triton.language.extra.tlx as tlx
    return {
        tlx.async_tasks: visit_withAsyncTasks,
        tlx.async_task: visit_withAsyncTask,
    }


class _LazyDispatch(dict):
    _initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self._initialized = True
            try:
                self.update(_build_dispatch())
            except (ImportError, AttributeError):
                pass

    def __getitem__(self, key):
        self._ensure_initialized()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_initialized()
        return super().__contains__(key)

    def get(self, key, default=None):
        self._ensure_initialized()
        return super().get(key, default)

    def items(self):
        self._ensure_initialized()
        return super().items()


TLX_WITH_DISPATCH = _LazyDispatch()
