"""Optional P0–P5.18 handoff integration, loaded through SGLang entry points."""


def register():
    import os

    # Installing the package alone must not change ordinary serving processes.
    if not os.environ.get("HANDOFF_CTL_DIR"):
        return

    from .hooks import register_hooks

    try:
        register_hooks()
    except Exception as exc:
        # The upstream loader logs Exception and continues. An explicitly
        # enabled handoff deployment must not serve with only partial hooks.
        raise SystemExit(f"remote_attn plugin initialization failed: {exc}") from exc
