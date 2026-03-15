import builtins

try:
    profile = builtins.profile # type: ignore[attr-defined]
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
