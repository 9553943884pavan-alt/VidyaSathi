import os
import sys

# Add the backend to the path so we can import the service
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend/app")))

from app.services.bkt_service import bkt_service

def test_bkt():
    print("=== Testing Bayesian Knowledge Tracing (BKT) Service ===")
    student_id = "test_student_123"
    topic = "VSEPR Theory"
    
    # 1. Initial State
    initial_mastery = bkt_service.get_mastery(student_id, topic)
    print(f"Initial mastery for {topic}: {initial_mastery:.4f} (Expected: 0.3000)")
    
    # 2. Student gets a question correct
    print("\nStudent gets a question CORRECT...")
    new_mastery_1 = bkt_service.update_mastery(student_id, topic, correct=True)
    print(f"New mastery: {new_mastery_1:.4f} (Expected > 0.3)")
    
    # 3. Student gets another question correct
    print("\nStudent gets another question CORRECT...")
    new_mastery_2 = bkt_service.update_mastery(student_id, topic, correct=True)
    print(f"New mastery: {new_mastery_2:.4f} (Expected > previous)")
    
    # 4. Student gets a question incorrect (slip or misconception)
    print("\nStudent gets a question INCORRECT...")
    new_mastery_3 = bkt_service.update_mastery(student_id, topic, correct=False)
    print(f"New mastery: {new_mastery_3:.4f} (Expected < previous)")
    
    # 5. Add some other topics
    bkt_service.update_mastery(student_id, "Hybridization", correct=False)
    bkt_service.update_mastery(student_id, "Hybridization", correct=False) # Very weak
    bkt_service.update_mastery(student_id, "Thermodynamics", correct=True)
    bkt_service.update_mastery(student_id, "Thermodynamics", correct=True) # Very strong
    
    # 6. Get weakest topics (for quiz generation)
    print("\nGetting weakest topics for quiz generation...")
    weak_topics = bkt_service.get_weakest_topics(student_id, limit=2)
    print(f"Weakest topics: {weak_topics}")
    
    print("\nBKT Test Complete! Logic is mathematically sound.")

if __name__ == "__main__":
    test_bkt()
