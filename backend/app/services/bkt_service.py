import sqlite3
from typing import Dict, Optional

class BKTModel:
    def __init__(self, p_known_init: float = 0.3, p_learn: float = 0.1, p_guess: float = 0.2, p_slip: float = 0.1):
        """
        Bayesian Knowledge Tracing Model
        p_known_init: Initial probability of knowing the topic
        p_learn: Probability of learning the topic after an opportunity
        p_guess: Probability of answering correctly without knowing the topic
        p_slip: Probability of answering incorrectly despite knowing the topic
        """
        self.p_known_init = p_known_init
        self.p_learn = p_learn
        self.p_guess = p_guess
        self.p_slip = p_slip

    def update_p_known(self, p_known_prev: float, correct: bool) -> float:
        """
        Updates the probability of knowing the topic given a correct/incorrect response.
        """
        if correct:
            # P(L|Correct) = [P(Correct|L) * P(L)] / P(Correct)
            # P(Correct|L) = 1 - p_slip
            # P(Correct) = P(Correct|L)*P(L) + P(Correct|~L)*P(~L) = (1 - p_slip)*P(L) + p_guess*(1 - P(L))
            p_correct = (1 - self.p_slip) * p_known_prev + self.p_guess * (1 - p_known_prev)
            p_known_given_obs = ((1 - self.p_slip) * p_known_prev) / p_correct if p_correct > 0 else 0
        else:
            # P(L|Incorrect) = [P(Incorrect|L) * P(L)] / P(Incorrect)
            # P(Incorrect|L) = p_slip
            # P(Incorrect) = P(Incorrect|L)*P(L) + P(Incorrect|~L)*P(~L) = p_slip*P(L) + (1 - p_guess)*(1 - P(L))
            p_incorrect = self.p_slip * p_known_prev + (1 - self.p_guess) * (1 - p_known_prev)
            p_known_given_obs = (self.p_slip * p_known_prev) / p_incorrect if p_incorrect > 0 else 0

        # Account for learning during the opportunity
        # P(L_new) = P(L_given_obs) + (1 - P(L_given_obs)) * p_learn
        p_known_new = p_known_given_obs + (1 - p_known_given_obs) * self.p_learn
        return p_known_new

class BKTService:
    def __init__(self, db_path: str = "vidya_sathi.db"):
        self.db_path = db_path
        self.model = BKTModel()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bkt_mastery (
                    student_id TEXT,
                    topic TEXT,
                    p_known REAL,
                    PRIMARY KEY (student_id, topic)
                )
            ''')
            conn.commit()

    def get_mastery(self, student_id: str, topic: str) -> float:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT p_known FROM bkt_mastery WHERE student_id = ? AND topic = ?',
                (student_id, topic)
            )
            row = cursor.fetchone()
            if row:
                return row[0]
            return self.model.p_known_init

    def update_mastery(self, student_id: str, topic: str, correct: bool) -> float:
        p_known_prev = self.get_mastery(student_id, topic)
        p_known_new = self.model.update_p_known(p_known_prev, correct)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO bkt_mastery (student_id, topic, p_known)
                VALUES (?, ?, ?)
                ON CONFLICT(student_id, topic) DO UPDATE SET p_known = excluded.p_known
            ''', (student_id, topic, p_known_new))
            conn.commit()
            
        return p_known_new

    def get_weakest_topics(self, student_id: str, limit: int = 3) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT topic FROM bkt_mastery WHERE student_id = ? ORDER BY p_known ASC LIMIT ?',
                (student_id, limit)
            )
            return [row[0] for row in cursor.fetchall()]

# Singleton instance
bkt_service = BKTService()
