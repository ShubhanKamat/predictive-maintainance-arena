

"""
FastAPI server for the Predictive Maintenance Arena.
Uses OpenEnv's create_app for proper WebSocket + HTTP support.
"""

import os
import sys

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_app

try:
    from ..models import MaintenanceAction, MaintenanceObservation
    from .environment import MaintenanceArenaEnvironment
except ImportError:
    from models import MaintenanceAction, MaintenanceObservation
    from server.environment import MaintenanceArenaEnvironment


app = create_app(
    MaintenanceArenaEnvironment,
    MaintenanceAction,
    MaintenanceObservation,
    env_name="pred_maint_arena",
)


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
