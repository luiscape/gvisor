"""
vllm_sleep_server.py — vLLM OpenAI API server + /sleep and /wake_up endpoints.

Drop-in replacement for `python3 -m vllm.entrypoints.openai.api_server`.
Adds three endpoints for GPU checkpoint/restore lifecycle management:

    POST /sleep?level=0    Pause scheduling (CUDA idle, memory retained)
    POST /sleep?level=1    Offload weights to CPU, discard KV cache
    POST /sleep?level=2    Discard ALL GPU memory
    POST /wake_up          Resume from sleep
    GET  /is_sleeping      Check sleep state

Checkpoint flow with gVisor's native cuda-checkpoint integration:
    1. curl -X POST localhost:8000/sleep?level=0   # quiesce scheduler
    2. runsc checkpoint --cuda-checkpoint-path=/usr/local/bin/cuda-checkpoint
       (the sentry runs cuda-checkpoint --toggle on every CUDA process,
        then serializes; after restore it toggles them back automatically)
    3. runsc restore
    4. curl -X POST localhost:8000/wake_up          # resume scheduler

This is a copy of gcr/test/vllm_sleep_patch.py (known-working with
vLLM 0.18), decoupled from the GCR/libgcr project.
"""

import asyncio
import sys


def add_sleep_endpoints(app, get_engine_fn):
    """Add /sleep, /wake_up, /is_sleeping routes to a FastAPI app."""

    from fastapi import Query
    from fastapi.responses import JSONResponse

    @app.post("/sleep")
    async def sleep_endpoint(
        level: int = Query(default=1, ge=0, le=2),
        mode: str = Query(default="abort"),
    ):
        engine = get_engine_fn()
        if engine is None:
            return JSONResponse(
                status_code=503,
                content={"error": "engine not ready"},
            )
        try:
            await engine.sleep(level=level, mode=mode)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )
        return {"status": "ok", "sleeping": True, "level": level, "mode": mode}

    @app.post("/wake_up")
    async def wake_up_endpoint():
        engine = get_engine_fn()
        if engine is None:
            return JSONResponse(
                status_code=503,
                content={"error": "engine not ready"},
            )
        try:
            await engine.wake_up()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )
        return {"status": "ok", "sleeping": False}

    @app.get("/is_sleeping")
    async def is_sleeping_endpoint():
        engine = get_engine_fn()
        if engine is None:
            return JSONResponse(
                status_code=503,
                content={"error": "engine not ready"},
            )
        try:
            sleeping = await engine.is_sleeping()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )
        return {"status": "ok", "sleeping": sleeping}

    print("[cr-bench] Added /sleep, /wake_up, /is_sleeping endpoints", flush=True)


def main():
    from vllm.entrypoints.openai import api_server as mod

    # Engine reference — resolved lazily after server startup.
    # vLLM 0.18 stores the engine in app.state.engine_client (set by
    # init_app_state during run_server).  We also check module-level
    # attributes as a fallback for other vLLM versions.
    _app_ref = [None]  # mutable ref so the closure can update it

    def get_engine():
        # Primary: vLLM 0.18 stores engine in app.state.engine_client
        a = _app_ref[0] or getattr(mod, "app", None)
        if a is not None:
            state = getattr(a, "state", None)
            if state is not None:
                ec = getattr(state, "engine_client", None)
                if ec is not None and hasattr(ec, "sleep"):
                    return ec
        # Fallback: check module-level attributes
        for attr in ("async_llm", "engine", "llm_engine", "engine_client"):
            obj = getattr(mod, attr, None)
            if obj is not None and hasattr(obj, "sleep"):
                return obj
        return None

    # In vLLM 0.18, the FastAPI app is created inside run_server /
    # build_app, not at module level.  We wrap run_server to inject
    # our endpoints after the app is created but before serving starts.
    _orig_run_server = getattr(mod, "run_server", None)

    if _orig_run_server is not None:

        async def patched_run_server(args):
            # The original run_server calls build_app() which creates
            # the FastAPI app and stores it + the engine in module globals.
            # We need to intercept AFTER build_app but BEFORE uvicorn.
            #
            # Strategy: monkeypatch serve_http to grab the app before serving.
            _orig_serve = getattr(mod, "serve_http", None)

            async def patched_serve(app, **kwargs):
                _app_ref[0] = app  # capture for get_engine
                if not hasattr(app, "_cr_bench_patched"):
                    add_sleep_endpoints(app, get_engine)
                    app._cr_bench_patched = True
                    print("[cr-bench] Endpoints injected into app", flush=True)
                return await _orig_serve(app, **kwargs)

            if _orig_serve is not None:
                mod.serve_http = patched_serve

            return await _orig_run_server(args)

        mod.run_server = patched_run_server

    # Build the argument parser and run.
    # vLLM 0.18 has FlexibleArgumentParser in the api_server module itself.
    FlexibleArgumentParser = getattr(mod, "FlexibleArgumentParser", None)
    make_arg_parser_fn = getattr(mod, "make_arg_parser", None)
    run_server_fn = getattr(mod, "run_server", None)  # our patched version

    if FlexibleArgumentParser and make_arg_parser_fn and run_server_fn:
        try:
            parser = make_arg_parser_fn(FlexibleArgumentParser())
            args = parser.parse_args(sys.argv[1:])
            print("[cr-bench] Using make_arg_parser + run_server", flush=True)
            asyncio.run(run_server_fn(args))
            return
        except Exception as e:
            print(f"[cr-bench] make_arg_parser + run_server failed: {e}", flush=True)

    # Fallback: run the module directly (won't have sleep endpoints).
    print("[cr-bench] WARNING: using runpy fallback (no sleep endpoints)", flush=True)
    import runpy

    sys.argv[0] = "vllm.entrypoints.openai.api_server"
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
