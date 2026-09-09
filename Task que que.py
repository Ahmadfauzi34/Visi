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

class RobustSinkhornQueue:
    def __init__(self, db_path: str):
        self.db_path = db_path

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
                SET status = 'PENDING', locked_by = NULL, locked_until = NULL, heartbeat_at = NULL, updated_at = ?
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
        # Ambil data tanpa menahan transaksi basis data
        def _fetch_op(conn):
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, task_name, task_type, priority, (? - created_at) as age
                FROM tasks WHERE status = 'PENDING' AND scheduled_at <= ?
                ORDER BY priority DESC, id ASC LIMIT ?;
            """, (now, now, total_available))
            return cursor.fetchall()
        
        tasks = DatabaseManager.execute_with_retry(self.db_path, _fetch_op)
        if not tasks: 
            return []

        N = len(tasks)
        M = len(active_workers)
        cost_matrix = np.zeros((N, M))
        
        INF_PENALTY = 1e4  # Hard-barrier: mencegah pemaksaan tipe mesin yang tidak kompatibel

        for i, (t_id, t_name, t_type, priority, age) in enumerate(tasks):
            prio_weight = 1.0 + (max(priority, 0) * 0.15)  # Proteksi batas bawah non-negatif
            for j, worker in enumerate(active_workers):
                if t_type == "gpu":
                    affinity = 0.0 if worker.worker_type == "gpu" else INF_PENALTY
                elif t_type == "cpu":
                    affinity = 0.0 if worker.worker_type == "cpu" else 8.0
                else:
                    affinity = 2.0
                
                cost_matrix[i, j] = (affinity * prio_weight) + 1.0

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
            # Validasi hard constraint: batalkan jika solver memaksakan tugas ke mesin tidak kompatibel
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

        # Update atomik: hanya ubah jika status saat eksekusi masih PENDING
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
                SELECT id, task_name, task_type, payload, retry_count, max_retries
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
                        "payload": row[3], "retry_count": row[4], "max_retries": row[5]
                    }
            return None
        return DatabaseManager.execute_with_retry(self.db_path, _op)

    def heartbeat(self, task_id: int, worker_id: str, extend_sec: float = 10.0) -> bool:
        def _op(conn):
            now = time.time()
            lease_until = now + extend_sec
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tasks 
                SET locked_until = ?, heartbeat_at = ?, updated_at = ? 
                WHERE id = ? AND locked_by = ? AND status = 'RUNNING';
            """, (lease_until, now, now, task_id, worker_id))
            return cursor.rowcount > 0
        return DatabaseManager.execute_with_retry(self.db_path, _op)

    def complete_task(self, task_id: int, worker_id: str) -> bool:
        def _op(conn):
            now = time.time()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tasks 
                SET status = 'COMPLETED', locked_by = NULL, locked_until = NULL, heartbeat_at = NULL, updated_at = ? 
                WHERE id = ? AND locked_by = ? AND status = 'RUNNING';
            """, (now, task_id, worker_id))
            return cursor.rowcount > 0
        return DatabaseManager.execute_with_retry(self.db_path, _op)

    def fail_task(self, task_id: int, worker_id: str, error_msg: str, max_backoff: float = 300.0) -> bool:
        def _op(conn):
            now = time.time()
            cursor = conn.cursor()
            # Dukung kegagalan saat status ASSIGNED maupun RUNNING
            cursor.execute("""
                SELECT retry_count, max_retries 
                FROM tasks 
                WHERE id = ? AND locked_by = ? AND status IN ('ASSIGNED', 'RUNNING');
            """, (task_id, worker_id))
            row = cursor.fetchone()
            if not row:
                return False  # Kunci hilang atau sewa telah kedaluwarsa

            current_retries, db_max_retries = row[0], row[1]
            next_retry = current_retries + 1

            if next_retry >= db_max_retries:
                cursor.execute("""
                    UPDATE tasks 
                    SET status = 'FAILED', retry_count = ?, locked_by = NULL, locked_until = NULL, heartbeat_at = NULL, error_log = ?, updated_at = ? 
                    WHERE id = ? AND locked_by = ?;
                """, (next_retry, error_msg, now, task_id, worker_id))
            else:
                # Backoff eksponensial dengan jitter acak dan batas maksimum
                backoff_seconds = min((2 ** current_retries) * 2.0 + random.uniform(0.5, 2.0), max_backoff)
                scheduled_next = now + backoff_seconds
                cursor.execute("""
                    UPDATE tasks 
                    SET status = 'PENDING', retry_count = ?, locked_by = NULL, locked_until = NULL, heartbeat_at = NULL, scheduled_at = ?, error_log = ?, updated_at = ? 
                    WHERE id = ? AND locked_by = ?;
                """, (next_retry, scheduled_next, error_msg, now, task_id, worker_id))
            return cursor.rowcount > 0
        return DatabaseManager.execute_with_retry(self.db_path, _op)
