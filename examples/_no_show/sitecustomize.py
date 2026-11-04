# Turn plotly's fig.show() into a no-op when this folder is on PYTHONPATH
try:
    import plotly.basedatatypes as _b

    _b.BaseFigure.show = lambda self, *args, **kwargs: None
except ImportError:
    pass
