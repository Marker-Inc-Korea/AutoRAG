import sqlite3
import json
from typing import Optional, List
from datetime import datetime
from src.schema import Trial, TrialConfig

class SQLiteTrialDB:
	def __init__(self, db_path: str):
		self.db_path = db_path
		self.init_db()

	def init_db(self):
		"""데이터베이스 초기화 및 테이블 생성"""
		with sqlite3.connect(self.db_path) as conn:
			conn.execute("""
				CREATE TABLE IF NOT EXISTS trials (
					id TEXT PRIMARY KEY,
					project_id TEXT NOT NULL,
					name TEXT,
					status TEXT,
					config JSON,
					created_at TEXT,
					report_task_id TEXT,
					chat_task_id TEXT
				)
			""")
			# 인덱스 생성
			conn.execute("CREATE INDEX IF NOT EXISTS idx_project_id ON trials(project_id)")

	def get_trial(self, trial_id: str) -> Optional[Trial]:
		"""특정 trial 조회"""
		with sqlite3.connect(self.db_path) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.execute("SELECT * FROM trials WHERE id = ?", (trial_id,))
			row = cursor.fetchone()
			
			if row:
				trial_dict = dict(row)
				if trial_dict['config']:
					trial_dict['config'] = TrialConfig.model_validate_json(trial_dict['config'])
				if trial_dict['created_at']:
					trial_dict['created_at'] = datetime.fromisoformat(trial_dict['created_at'])
				return Trial(**trial_dict)
			return None

	def set_trial(self, trial: Trial):
		"""trial 저장 또는 업데이트"""
		with sqlite3.connect(self.db_path) as conn:
			conn.execute("""
				INSERT OR REPLACE INTO trials 
				(id, project_id, name, status, config, created_at, report_task_id, chat_task_id)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?)
			""", (
				trial.id,
				trial.project_id,
				trial.name,
				trial.status,
				trial.config.model_dump_json() if trial.config else None,
				trial.created_at.isoformat() if trial.created_at else None,
				trial.report_task_id,
				trial.chat_task_id
			))

	def set_trial_config(self, trial_id: str, config: TrialConfig):
		"""trial config 업데이트"""
		with sqlite3.connect(self.db_path) as conn:
			conn.execute("""
				UPDATE trials 
				SET config = ?
				WHERE id = ?
			""", (config.model_dump_json(), trial_id))

	def get_all_config_ids(self) -> List[str]:
		"""모든 trial의 config ID 목록 조회"""
		with sqlite3.connect(self.db_path) as conn:
			cursor = conn.execute("""
				SELECT DISTINCT id 
				FROM trials 
				WHERE config IS NOT NULL
				ORDER BY created_at DESC
			""")
			return [row[0] for row in cursor.fetchall()]

	def get_all_trial_ids(self, project_id: Optional[str] = None) -> List[str]:
		"""모든 trial ID 조회 (프로젝트별 필터링 가능)"""
		with sqlite3.connect(self.db_path) as conn:
			if project_id:
				cursor = conn.execute("""
					SELECT id 
					FROM trials 
					WHERE project_id = ?
					ORDER BY created_at DESC
				""", (project_id,))
			else:
				cursor = conn.execute("""
					SELECT id 
					FROM trials
					ORDER BY created_at DESC
				""")
			return [row[0] for row in cursor.fetchall()]

	def delete_trial(self, trial_id: str):
		"""trial 삭제"""
		with sqlite3.connect(self.db_path) as conn:
			conn.execute("DELETE FROM trials WHERE id = ?", (trial_id,))

	def get_trials_by_project(self, project_id: str, limit: int = 10, offset: int = 0) -> List[Trial]:
		"""프로젝트별 trial 목록 조회 (페이지네이션)"""
		with sqlite3.connect(self.db_path) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.execute("""
				SELECT * FROM trials 
				WHERE project_id = ? 
				ORDER BY created_at DESC
				LIMIT ? OFFSET ?
			""", (project_id, limit, offset))
			
			trials = []
			for row in cursor.fetchall():
				trial_dict = dict(row)
				if trial_dict['config']:
					trial_dict['config'] = TrialConfig(**json.loads(trial_dict['config']))
				if trial_dict['created_at']:
					trial_dict['created_at'] = datetime.fromisoformat(trial_dict['created_at'])
				trials.append(Trial(**trial_dict))
			return trials
