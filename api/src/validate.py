import os
from functools import wraps

from quart import jsonify

from src.schema import Trial
from src.trial_config import PandasTrialDB


def project_exists(work_dir):
	def decorator(func):
		@wraps(func)
		async def wrapper(*args, **kwargs):
			# Get project_id from request arguments
			project_id = kwargs.get('project_id')

			if not project_id:
				return jsonify({
					'error': 'project_id is required'
				}), 400

			# Check if project directory exists
			project_path = os.path.join(work_dir, project_id)
			if not os.path.exists(project_path):
				return jsonify({
					'error': f'Project with id {project_id} does not exist'
				}), 404

			# If everything is okay, proceed with the endpoint function
			return await func(*args, **kwargs)

		return wrapper
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

			trial_config_path = os.path.join(work_dir, project_id, "trial_config.csv")
			trial_config_db = PandasTrialDB(trial_config_path)
			trial = trial_config_db.get_trial(trial_id)
			if trial is None or not isinstance(trial, Trial):
				return jsonify({
					'error': f'Trial with id {trial_id} does not exist'
				}), 404

			return await func(*args, **kwargs)
		return wrapper
	return decorator
