"""
FastAPI application entry-point.

* Configures CORS middleware (all origins allowed for development).
* Registers all API routers.
* Runs startup logic: database table creation & model pre-loading.
* Optionally serves static model figures from ``ml/figures/``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, FIGURES_DIR
from app.database import create_tables
from app.routes import health, mining, performance, predict, stats
from app.services.prediction_service import preload_all_models

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executed once on application startup and shutdown."""
    # ── Startup ─────────────────────────────────────────────────────
    logger.info("Starting %s v%s …", APP_TITLE, APP_VERSION)

    # 1. Create database tables (idempotent)
    create_tables()
    logger.info("Database tables verified.")

    # 2. Pre-load ML models & scaler
    try:
        preload_all_models()
    except Exception as exc:
        logger.warning(
            "Could not pre-load models (they may not be trained yet): %s", exc
        )

    # 3. Mount figures directory if it exists
    if FIGURES_DIR.exists():
        app.mount(
            "/figures",
            StaticFiles(directory=str(FIGURES_DIR)),
            name="figures",
        )
        logger.info("Serving static figures from %s", FIGURES_DIR)
    else:
        logger.info("Figures directory not found at %s – skipping mount.", FIGURES_DIR)

    logger.info("🚀 %s is ready!", APP_TITLE)
    yield
    # ── Shutdown ────────────────────────────────────────────────────
    logger.info("Shutting down %s …", APP_TITLE)


# ── App instance ───────────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

# ── CORS (allow all origins for dev) ───────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(stats.router)
app.include_router(performance.router)
app.include_router(mining.router)
app.include_router(health.router)


# ── Root redirect to docs ──────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the interactive API docs."""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")
