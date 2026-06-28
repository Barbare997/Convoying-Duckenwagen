"""Flask routes for live intersection turn tuning (project follower UI)."""

from flask import jsonify, request

from tasks.project.packages import agent


def register_intersection_turn_routes(app):
    @app.route("/get_intersection_turn")
    def get_intersection_turn():
        return jsonify(agent.get_intersection_turn_params())

    @app.route("/update_intersection_turn", methods=["POST"])
    def update_intersection_turn():
        data = request.json or {}
        try:
            return jsonify(agent.patch_intersection_turn_params(data))
        except (TypeError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except OSError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/intersection/test_turn", methods=["POST"])
    def intersection_test_turn():
        data = request.json or {}
        try:
            result = agent.request_follower_test_turn(data.get("direction", ""))
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        if result.get("status") == "error":
            return jsonify(result), 400
        return jsonify(result)
