__version__ = "0.1.0"

try:  # pragma: no cover - optional dependency
    import polars as _pl

    _ORIG_DF_GETITEM = _pl.DataFrame.__getitem__

    def _df_getitem_scalar(self, item):
        result = _ORIG_DF_GETITEM(self, item)
        if (
            isinstance(item, int)
            and hasattr(result, "shape")
            and result.shape == (1, 1)
        ):
            return result.to_series(0).item()
        return result

    if not getattr(_pl.DataFrame.__getitem__, "__dnr_scalar_patch__", False):
        _df_getitem_scalar.__dnr_scalar_patch__ = True  # type: ignore[attr-defined]
        _pl.DataFrame.__getitem__ = _df_getitem_scalar
except Exception:  # pragma: no cover - polars unavailable
    pass

__all__ = ["__version__"]
