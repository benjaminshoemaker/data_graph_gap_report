__version__ = "0.1.2"

try:  # pragma: no cover - optional dependency
    import polars as _pl

    _ORIG_DF_GETITEM = _pl.DataFrame.__getitem__
    _ORIG_TO_DICT_METHOD = _pl.DataFrame.to_dict

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

    def _df_to_dicts_safe(self):
        data = _ORIG_TO_DICT_METHOD(self, as_series=False)
        columns = list(data.keys())
        if not columns:
            return []
        row_count = len(data[columns[0]])
        rows = []
        for idx in range(row_count):
            rows.append({col: data[col][idx] for col in columns})
        return rows

    if not getattr(_pl.DataFrame.to_dicts, "__dnr_safe_patch__", False):
        _df_to_dicts_safe.__dnr_safe_patch__ = True  # type: ignore[attr-defined]
        _pl.DataFrame.to_dicts = _df_to_dicts_safe
except Exception:  # pragma: no cover - polars unavailable
    pass

__all__ = ["__version__"]
