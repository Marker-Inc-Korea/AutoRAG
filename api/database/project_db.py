import sqlite3
import json
from typing import Optional, List
from datetime import datetime
from src.schema import Trial, TrialConfig

import os


class SQLiteProjectDB:
    def __init__(self, project_id: str):
        print(f"Initializing SQLiteProjectDB for project_id: {project_id}")
        self.project_id = project_id
        self.db_path = self._get_db_path()
        self._init_db()

    def _get_db_path(self) -> str:
        """프로젝트 ID로부터 DB 경로 생성"""
        # 1. 기본 작업 디렉토리 설정
        work_dir = os.getenv("WORK_DIR", "/app/projects")

        # 2. 절대 경로로 변환 (상대 경로 해결)
        work_dir = os.path.abspath(work_dir)

        # 3. 최종 DB 파일 경로 생성
        db_path = os.path.join(work_dir, self.project_id, "project.db")

        # 디버깅을 위한 로그
        print(f"WORK_DIR (raw): {os.getenv('WORK_DIR', '/app/projects')}")
        print(f"WORK_DIR (abs): {work_dir}")
        print(f"Project ID: {self.project_id}")
        print(f"Final DB path: {db_path}")

        return db_path

    def _init_db(self):
        """DB 초기화 (필요한 경우 디렉토리 및 테이블 생성)"""
        db_exists = os.path.exists(self.db_path)
        db_dir = os.path.dirname(self.db_path)

        print(f"DB Path: {self.db_path}")
        print(f"DB Directory: {db_dir}")
        print(f"DB exists: {db_exists}")
        print(f"Directory exists: {os.path.exists(db_dir)}")

        # DB 파일이 없을 때만 초기화 작업 수행
        if not db_exists:
            # 디렉토리가 없을 때만 생성
            if not os.path.exists(db_dir):
                print(f"Creating directory: {db_dir}")
                os.makedirs(db_dir)
                # 디렉토리 권한 설정 (777)
                os.chmod(db_dir, 0o777)

            try:
                print(f"Creating database: {self.db_path}")
                with sqlite3.connect(self.db_path) as conn:
                    print("Successfully connected to database")
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS trials (
                            id TEXT PRIMARY KEY,
                            project_id TEXT NOT NULL,
                            name TEXT,
                            status TEXT,
                            config JSON,
                            created_at TEXT,
                            report_task_id TEXT,
                            chat_task_id TEXT,
                            api_pid NUMERIC
                        )
                    """)
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_project_id ON trials(project_id)"
                    )

                # DB 파일 권한 설정 (666)
                os.chmod(self.db_path, 0o666)
            except Exception as e:
                print(f"Error creating database: {str(e)}")
                print(f"Current working directory: {os.getcwd()}")
                raise

    def get_trial(self, trial_id: str) -> Optional[Trial]:
        """특정 trial 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM trials WHERE id = ?", (trial_id,))
            row = cursor.fetchone()

            if row:
                trial_dict = dict(row)
                if trial_dict["config"]:
                    trial_dict["config"] = TrialConfig.model_validate_json(
                        trial_dict["config"]
                    )
                if trial_dict["created_at"]:
                    trial_dict["created_at"] = datetime.fromisoformat(
                        trial_dict["created_at"]
                    )
                return Trial(**trial_dict)
            return None

    def set_trial(self, trial: Trial):
        """trial 저장 또는 업데이트"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trials
                (id, project_id, name, status, config, created_at, report_task_id, chat_task_id, api_pid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			""",
                (
                    trial.id,
                    trial.project_id,
                    trial.name,
                    trial.status,
                    trial.config.model_dump_json() if trial.config else None,
                    trial.created_at.isoformat() if trial.created_at else None,
                    trial.report_task_id,
                    trial.chat_task_id,
                    trial.api_pid,
                ),
            )

    def set_trial_config(self, trial_id: str, config: TrialConfig):
        """trial config 업데이트"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
				UPDATE trials
				SET config = ?
				WHERE id = ?
			""",
                (config.model_dump_json(), trial_id),
            )

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

    def get_all_configs(self) -> List[TrialConfig]:
        """모든 trial의 config 목록 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT config
                FROM trials
                WHERE config IS NOT NULL
                ORDER BY created_at DESC
            """)
            return [
                TrialConfig.model_validate_json(row[0]) for row in cursor.fetchall()
            ]

    def get_all_trial_ids(self, project_id: Optional[str] = None) -> List[str]:
        """모든 trial ID 조회 (프로젝트별 필터링 가능)"""
        with sqlite3.connect(self.db_path) as conn:
            if project_id:
                cursor = conn.execute(
                    """
                    SELECT id
					FROM trials
					WHERE project_id = ?
					ORDER BY created_at DESC
				""",
                    (project_id,),
                )
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

    def get_trials_by_project(
        self, project_id: str, limit: int = 10, offset: int = 0
    ) -> List[Trial]:
        """프로젝트별 trial 목록 조회 (페이지네이션)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM trials
				WHERE project_id = ?
				ORDER BY created_at DESC
				LIMIT ? OFFSET ?
			""",
                (project_id, limit, offset),
            )

            trials = []
            for row in cursor.fetchall():
                trial_dict = dict(row)
                if trial_dict["config"]:
                    trial_dict["config"] = TrialConfig(
                        **json.loads(trial_dict["config"])
                    )
                if trial_dict["created_at"]:
                    trial_dict["created_at"] = datetime.fromisoformat(
                        trial_dict["created_at"]
                    )
                trials.append(Trial(**trial_dict))
        return trials
