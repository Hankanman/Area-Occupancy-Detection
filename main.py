"""Entry point for running the simulator backend locally."""

import os

from simulator.app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    host = os.getenv("FLASK_HOST", "0.0.0.0").strip()
    app.run(host=host, port=port, debug=debug)
