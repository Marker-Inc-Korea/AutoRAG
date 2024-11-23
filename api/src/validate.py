import os
from functools import wraps

from quart import jsonify

from src.schema import Trial
from src.trial_config import SQLiteTrialDB

def project_exists(base_dir: str):
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            project_id = kwargs.get('project_id')
            if not project_id:
                return jsonify({"error": "Project ID is required"}), 400

            project_dir = os.path.join(base_dir, project_id)
            if not os.path.exists(project_dir):
                return jsonify({"error": "Project not found"}), 404

            # SQLite DB 초기화 추가
            db_path = os.path.join(project_dir, "trials.db")
            SQLiteTrialDB(db_path)

            return await f(*args, **kwargs)
        return decorated_function
    return decorator

def trial_exists(work_dir: str):
	def decorator(func):
		@wraps(func)
		async def wrapper(*args, **kwargs):
			project_id = kwargs.get("project_id")
			trial_id = kwargs.get("trial_id")

			if not trial_id:
				return jsonify({
					'error': 'trial_id is required'
				}), 400

			trial_config_path = os.path.join(work_dir, project_id, "trials.db")
			trial_config_db = SQLiteTrialDB(trial_config_path)
			trial = trial_config_db.get_trial(trial_id)
			if trial is None or not isinstance(trial, Trial):
				return jsonify({
					'error': f'Trial with id {trial_id} does not exist'
				}), 404

			return await func(*args, **kwargs)
		return wrapper
	return decorator
