import os
from abc import ABCMeta, abstractmethod
from typing import Optional, List

import pandas as pd

from src.schema import TrialConfig, Trial


class BaseTrialDB(metaclass=ABCMeta):
    @abstractmethod
    def set_trial(self, trial: Trial):
        pass

    @abstractmethod
    def get_trial(self, trial_id: str) -> Optional[Trial]:
        pass

    @abstractmethod
    def set_trial_config(self, trial_id: str, config: TrialConfig):
        pass

    @abstractmethod
    def get_trial_config(self, trial_id: str) -> Optional[TrialConfig]:
        pass

    @abstractmethod
    def get_all_config_ids(self) -> List[str]:
        pass


class PandasTrialDB(BaseTrialDB):
    def __init__(self, df_path: str):
        self.columns = [
            "id",
            "project_id",
            "config",
            "name",
            "status",
            "created_at",
            "report_task_id",
            "chat_task_id",
        ]
        self.df_path = df_path
        if not os.path.exists(df_path):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(df_path, index=False)
        else:
            try:
                df = pd.read_csv(df_path)
            except Exception:
                df = pd.DataFrame(columns=self.columns)
        self.df = df

    def set_trial(self, trial: Trial):
        new_row = pd.DataFrame(
            {
                "id": [trial.id],
                "project_id": [trial.project_id],
                "config": [trial.config.model_dump_json()],
                "name": [trial.name],
                "status": [trial.status],
                "created_at": [trial.created_at],
                "report_task_id": [trial.report_task_id],
                "chat_task_id": [trial.chat_task_id],
            }
        )
        if len(self.df.loc[self.df["id"] == trial.id]) > 0:
            self.df = self.df.loc[self.df["id"] != trial.id]
        self.df = pd.concat([self.df, new_row])
        self.df.to_csv(self.df_path, index=False)

    def get_trial(self, trial_id: str) -> Optional[Trial]:
        matches = self.df[self.df["id"] == trial_id]
        if len(matches) < 1:
            return None
        row = matches.iloc[0]
        if row.empty:
            return None
        return Trial(
            id=row["id"],
            project_id=row["project_id"],
            config=TrialConfig.model_validate_json(row["config"]),
            name=row["name"],
            status=row["status"],
            created_at=row["created_at"],
            report_task_id=row["report_task_id"],
            chat_task_id=row["chat_task_id"],
        )

    def set_trial_config(self, trial_id: str, config: TrialConfig):
        config_dict = config.model_dump_json()
        self.df.loc[self.df["id"] == trial_id, "config"] = config_dict
        self.df.to_csv(self.df_path, index=False)

    def get_trial_config(self, trial_id: str) -> Optional[TrialConfig]:
        config_dict = self.df.loc[self.df["id"] == trial_id]["config"].tolist()
        if len(config_dict) < 1:
            return None

        config = TrialConfig.model_validate_json(config_dict[0])
        return config

    def get_all_config_ids(self) -> List[str]:
        return self.df["id"].tolist()
