"""Flask routes for live follower spacing tuning (project follower UI)."""

from flask import jsonify, request

from tasks.project.packages import agent


def register_spacing_routes(app):
    @app.route("/get_spacing")
    def get_spacing():
        return jsonify(agent.get_spacing_params())

    @app.route("/update_spacing", methods=["POST"])
    def update_spacing():
        data = request.json or {}
        try:
            return jsonify(agent.patch_spacing_params(data))
        except (TypeError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except OSError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500
