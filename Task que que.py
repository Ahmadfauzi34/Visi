from dataclasses import dataclass
from typing import List, Dict, Optional
import sqlite3
import time
import random
import numpy as np

@dataclass
class WorkerDescriptor:
    worker_id: str
    worker_type: str
    capacity: int
    available_slots: int

class DatabaseManager:
    @staticmethod
    def init_db(db_path: str):
        def _op(conn):
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    payload TEXT,
                    priority INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    retry_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'PENDING',
                    locked_by TEXT,
                    locked_until REAL,
                    lease_epoch INTEGER DEFAULT 0,
                    heartbeat_at REAL,
                    created_at REAL,
                    scheduled_at REAL,
                    updated_at REAL,
                    error_log TEXT
                );
            """)
        DatabaseManager.execute_with_retry(db_path, _op)

    @staticmethod
    def execute_with_retry(db_path: str, op_func, max_retries: int = 5):
        delay = 0.05
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(db_path, timeout=10.0) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    res = op_func(conn)
                    conn.commit()
                    return res
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise

def _logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out

def sinkhorn_knopp_log_domain(C: np.ndarray, r: np.ndarray, c: np.ndarray, epsilon: float = 1.5, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    N, M = C.shape
    u = np.zeros(N)
    v = np.zeros(M)
    K = -C / max(epsilon, 1e-5)

    for _ in range(max_iter):
        u_prev = u
        u = np.log(r + 1e-12) - _logsumexp(K + v[None, :], axis=1)
        v = np.log(c + 1e-12) - _logsumexp(K + u[:, None], axis=0)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    P = np.exp(K + u[:, None] + v[None, :])
    return P

def round_transport_plan_bounded(P: np.ndarray, capacities: List[int]) -> List[int]:
    N, M = P.shape
    assignments = [-1] * N
    remaining_caps = list(capacities)

    pairs = []
    for i in range(N):
        for j in range(M):
            pairs.append((P[i, j], i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_tasks = set()
    for prob, i, j in pairs:
        if i not in assigned_tasks and remaining_caps[j] > 0:
            assignments[i] = j
            assigned_tasks.add(i)
            remaining_caps[j] -= 1
            if len(assigned_tasks) == N:
                break

    return assignments

class RobustSinkhornQueue:
    def __init__(self, db_path: str):
        self.db_path = db_path
        DatabaseManager.init_db(self.db_path)

    def enqueue(self, task_name: str, task_type: str, payload: str, priority: int = 0, max_retries: int = 3) -> int:
        def _op(conn):
            cursor = conn.cursor()
            now = time.time()
            cursor.execute("""
                INSERT INTO tasks (task_name, task_type, payload, priority, max_retries, status, created_at, scheduled_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'PENDING', ?, ?, ?);
            """, (task_name, task_type, payload, priority, max_retries, now, now, now))
            return cursor.lastrowid
        return DatabaseManager.execute_with_retry(self.db_path, _op)

    def recover_expired_leases(self) -> int:
        def _op(conn):
            now = time.time()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tasks
                SET status = 'PENDING', locked_by = NULL, locked_until = NULL, heartbeat_at = NULL,
                    lease_epoch = lease_epoch + 1, updated_at = ?
                WHERE status IN ('ASSIGNED', 'RUNNING') AND locked_until < ?;
            """, (now, now))
            return cursor.rowcount
        return DatabaseManager.execute_with_retry(self.db_path, _op)

    def dispatch_batch(self, workers: List[WorkerDescriptor], epsilon: float = 1.5, lease_sec: float = 10.0) -> List[Dict]:
        active_workers = [w for w in workers if w.available_slots > 0]
        if not active_workers: 
            return []
        total_available = sum(w.available_slots for w in active_workers)

        now = time.time()
        def _fetch_op(conn):
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, task_name, task_type, priority, (? - created_at) as age
                FROM tasks WHERE status = 'PENDING' AND scheduled_at <= ?
                ORDER BY priority DESC, id ASC LIMIT ?;
            """, (now, now, total_available))
            return cursor.fetchall()
        
        all_candidate_tasks = DatabaseManager.execute_with_retry(self.db_path, _fetch_op)
        if not all_candidate_tasks:
            return []

        INF_PENALTY = 1e4
        
        # Filter candidate tasks: compute affinity matrix and ensure task has at least one compatible worker
        compatible_tasks = []
        cost_rows = []

        for t_id, t_name, t_type, priority, age in all_candidate_tasks:
            row_costs = []
            has_compatible = False
            prio_term = 1.0 / (1.0 + max(priority, 0) * 0.15)  # Higher priority -> lower cost multiplier

            for worker in active_workers:
                if t_type == "gpu":
                    affinity = 0.0 if worker.worker_type == "gpu" else INF_PENALTY
                elif t_type == "cpu":
                    affinity = 0.0 if worker.worker_type == "cpu" else 8.0
                else:
                    affinity = 2.0
                
                if affinity < INF_PENALTY:
                    has_compatible = True

                cost = affinity + prio_term
                row_costs.append(cost)

            if has_compatible:
                compatible_tasks.append((t_id, t_name, t_type, priority, age))
                cost_rows.append(row_costs)

        if not compatible_tasks:
            return []

        tasks = compatible_tasks
        N = len(tasks)
        M = len(active_workers)
        cost_matrix = np.array(cost_rows)

        r_supply = np.ones(N) / N
        c_demand = np.array([w.available_slots / total_available for w in active_workers])
        
        P = sinkhorn_knopp_log_domain(cost_matrix, r_supply, c_demand, epsilon=epsilon)
        capacities = [w.available_slots for w in active_workers]
        slot_assignments = round_transport_plan_bounded(P, capacities)

        lease_until = now + lease_sec
        dispatched_candidates = []
        update_payloads = []

        for i, (t_id, t_name, t_type, priority, _) in enumerate(tasks):
            w_idx = slot_assignments[i]
            if w_idx != -1 and cost_matrix[i, w_idx] < INF_PENALTY:
                chosen_worker = active_workers[w_idx]
                update_payloads.append((
                    'ASSIGNED', chosen_worker.worker_id, lease_until, now, now, t_id
                ))
                dispatched_candidates.append({
                    "task_id": t_id, "task_name": t_name, "task_type": t_type,
                    "priority": priority, "worker_id": chosen_worker.worker_id,
                    "transport_score": round(float(P[i, w_idx]), 4)
                })

        if not update_payloads:
            return []

        def _update_op(conn):
            cur = conn.cursor()
            confirmed_dispatched = []
            for payload, disp in zip(update_payloads, dispatched_candidates):
                cur.execute("""
                    UPDATE tasks
                    SET status = ?, locked_by = ?, locked_until = ?, heartbeat_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'PENDING';
                """, payload)
                if cur.rowcount > 0:
                    confirmed_dispatched.append(disp)
            return confirmed_dispatched

        return DatabaseManager.execute_with_retry(self.db_path, _update_op)

    def claim_task(self, worker_id: str) -> Optional[Dict]:
        def _op(conn):
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, task_name, task_type, payload, retry_count, max_retries, lease_epoch
                FROM tasks WHERE status = 'ASSIGNED' AND locked_by = ?
                ORDER BY priority DESC, id ASC LIMIT 1;
            """, (worker_id,))
            row = cursor.fetchone()
            if row:
                t_id = row[0]
                now = time.time()
                cursor.execute("""
                    UPDATE tasks 
                    SET status = 'RUNNING', heartbeat_at = ?, updated_at = ? 
                    WHERE id = ? AND status = 'ASSIGNED' AND locked_by = ?;
                """, (now, now, t_id, worker_id))
                if cursor.rowcount > 0:
                    return {
                        "id": row[0], "task_name": row[1], "task_type": row[2], 
                        "payload": row[3], "retry_count": row[4], "max_retries": row[5],
                        "lease_epoch": row[6]
                    }
            return None
        return DatabaseManager.execute_with_retry(self.db_path, _op)

    def heartbeat(self, task_id: int, worker_id: str, lease_epoch: Optional[int] = None, extend_sec: float = 10.0) -> bool:
        def _op(conn):
            now = time.time()
            lease_until = now + extend_sec
            cursor = conn.cursor()
            if lease_epoch is not None:
                cursor.execute("""
                    UPDATE tasks
                    SET locked_until = ?, heartbeat_at = ?, updated_at = ?
                    WHERE id = ? AND locked_by = ? AND status = 'RUNNING' AND lease_epoch = ?;
                """, (lease_until, now, now, task_id, worker_id, lease_epoch))
            else:
                cursor.execute("""
                    UPDATE tasks
                    SET locked_until = ?, heartbeat_at = ?, updated_at = ?
                    WHERE id = ? AND locked_by = ? AND status = 'RUNNING';
                """, (lease_until, now, now, task_id, worker_id))
            return cursor.rowcount > 0
        return DatabaseManager.execute_with_retry(self.db_path, _op)

    def complete_task(self, task_id: int, worker_id: str, lease_epoch: Optional[int] = None) -> bool:
        def _op(conn):
            now = time.time()
            cursor = conn.cursor()
            if lease_epoch is not None:
                cursor.execute("""
                    UPDATE tasks
                    SET status = 'COMPLETED', locked_by = NULL, locked_until = NULL, heartbeat_at = NULL, updated_at = ?
                    WHERE id = ? AND locked_by = ? AND status = 'RUNNING' AND lease_epoch = ?;
                """, (now, task_id, worker_id, lease_epoch))
            else:
                cursor.execute("""
                    UPDATE tasks
                    SET status = 'COMPLETED', locked_by = NULL, locked_until = NULL, heartbeat_at = NULL, updated_at = ?
                    WHERE id = ? AND locked_by = ? AND status = 'RUNNING';
                """, (now, task_id, worker_id))
            return cursor.rowcount > 0
        return DatabaseManager.execute_with_retry(self.db_path, _op)

    def fail_task(self, task_id: int, worker_id: str, error_msg: str, lease_epoch: Optional[int] = None, max_backoff: float = 300.0) -> bool:
        def _op(conn):
            now = time.time()
            cursor = conn.cursor()
            if lease_epoch is not None:
                cursor.execute("""
                    SELECT retry_count, max_retries
                    FROM tasks
                    WHERE id = ? AND locked_by = ? AND status IN ('ASSIGNED', 'RUNNING') AND lease_epoch = ?;
                """, (task_id, worker_id, lease_epoch))
            else:
                cursor.execute("""
                    SELECT retry_count, max_retries
                    FROM tasks
                    WHERE id = ? AND locked_by = ? AND status IN ('ASSIGNED', 'RUNNING');
                """, (task_id, worker_id))
            row = cursor.fetchone()
            if not row:
                return False

            current_retries, db_max_retries = row[0], row[1]
            next_retry = current_retries + 1

            if next_retry >= db_max_retries:
                cursor.execute("""
                    UPDATE tasks 
                    SET status = 'FAILED', retry_count = ?, locked_by = NULL, locked_until = NULL, heartbeat_at = NULL, error_log = ?, updated_at = ? 
                    WHERE id = ? AND locked_by = ?;
                """, (next_retry, error_msg, now, task_id, worker_id))
            else:
                backoff_seconds = min((2 ** current_retries) * 2.0 + random.uniform(0.5, 2.0), max_backoff)
                scheduled_next = now + backoff_seconds
                cursor.execute("""
                    UPDATE tasks 
                    SET status = 'PENDING', retry_count = ?, locked_by = NULL, locked_until = NULL, heartbeat_at = NULL, scheduled_at = ?, error_log = ?, updated_at = ? 
                    WHERE id = ? AND locked_by = ?;
                """, (next_retry, scheduled_next, error_msg, now, task_id, worker_id))
            return cursor.rowcount > 0
        return DatabaseManager.execute_with_retry(self.db_path, _op)
