import pandas as pd
import json
from typing import Optional, List
from src.schema import Trial, TrialConfig
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, datetime):
			return obj.isoformat()
		return super().default(obj)

class PandasTrialDB:
	def __init__(self, path: str):
		self.path = path
		self.columns = [
			'id',
			'project_id',
			'config',
			'name',
			'status',
			'created_at',
			'report_task_id',
			'chat_task_id'
		]
		try:
			self.df = pd.read_csv(self.path)
			for col in self.columns:
				if col not in self.df.columns:
					self.df[col] = None
		except FileNotFoundError:
			self.df = pd.DataFrame(columns=self.columns)
			self.df.to_csv(self.path, index=False)

	def get_trial(self, trial_id: str) -> Optional[Trial]:
		try:
			trial_row = self.df[self.df['id'] == trial_id].iloc[0]
			trial_dict = trial_row.to_dict()
			
			# config 문자열을 딕셔너리로 변환
			if isinstance(trial_dict['config'], str):
				config_dict = json.loads(trial_dict['config'])
				# config 필드만 따로 처리
				if 'config' in config_dict:
					trial_dict['config'] = TrialConfig(**config_dict['config'])
				else:
					trial_dict['config'] = None
			
			# created_at을 datetime으로 변환
			if isinstance(trial_dict['created_at'], str):
				trial_dict['created_at'] = datetime.fromisoformat(trial_dict['created_at'])
			
			return Trial(**trial_dict)
		except (IndexError, KeyError, json.JSONDecodeError):
			return None

	def set_trial(self, trial: Trial):
		try:
			row_data = {
				'id': trial.id,
				'project_id': trial.project_id,
				'config': json.dumps(trial.model_dump(), cls=DateTimeEncoder),
				'name': trial.name,
				'status': trial.status,
				'created_at': trial.created_at.isoformat() if trial.created_at else None,
				'report_task_id': None,
				'chat_task_id': None
			}
			
			if self.df.empty or trial.id not in self.df['id'].values:
				# 새 행 추가
				new_row = pd.DataFrame([row_data], columns=self.columns)
				self.df = pd.concat([self.df, new_row], ignore_index=True)
			else:
				# 기존 행 업데이트
				idx = self.df.index[self.df['id'] == trial.id][0]
				for col in self.columns:
					self.df.at[idx, col] = row_data[col]
			
			# 변경사항 저장
			self.df.to_csv(self.path, index=False)
		except Exception as e:
			print(f"Error in set_trial: {e}")
			print(f"Row data: {row_data}")
			raise

	def get_trial_config(self, trial_id: str) -> Optional[Trial]:
		return self.get_trial(trial_id)

	def set_trial_config(self, trial_id: str, config: Trial):
		return self.set_trial(config)

	def get_all_trial_ids(self) -> List[str]:
		return self.df['id'].tolist()
