"""
Engine & Session Singleton Lifecycle Verification
Financial Market Regime & Event Intelligence Engine
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.database.connection import get_engine, get_session, get_database_url


def test_engine_singleton():
    print("==========================================")
    print("TESTING ENGINE & SESSION SINGLETON LIFECYCLE")
    print("==========================================")
    
    # 1. Repeated calls to get_engine() must return identical Engine reference
    engine1 = get_engine()
    engine2 = get_engine()
    engine3 = get_engine()
    
    print(f"Engine 1 ID: {id(engine1)}")
    print(f"Engine 2 ID: {id(engine2)}")
    print(f"Engine 3 ID: {id(engine3)}")
    
    assert engine1 is engine2, "FAILED: get_engine() returned different instances!"
    assert engine2 is engine3, "FAILED: get_engine() returned different instances!"
    print("[OK] get_engine() singleton verification passed! All calls return identical Engine object.\n")
    
    # 2. Check engine connection pool configuration
    pool = engine1.pool
    print("Engine Connection Pool Settings:")
    print(f"  - pool_size: {pool.size()}")
    print(f"  - max_overflow: {pool._max_overflow}")
    print(f"  - pre_ping: {engine1.pool._pre_ping}")
    
    assert pool.size() == 10, f"Expected pool_size=10, got {pool.size()}"
    assert pool._max_overflow == 20, f"Expected max_overflow=20, got {pool._max_overflow}"
    assert engine1.pool._pre_ping is True, "Expected pool_pre_ping=True"
    print("[OK] Connection pool settings verified (pool_size=10, max_overflow=20, pre_ping=True)!\n")
    
    # 3. Verify get_session() binds to shared singleton engine
    session1 = get_session()
    session2 = get_session()
    
    print(f"Session 1 bind engine ID: {id(session1.get_bind())}")
    print(f"Session 2 bind engine ID: {id(session2.get_bind())}")
    
    assert session1.get_bind() is engine1, "FAILED: Session 1 is not bound to shared engine!"
    assert session2.get_bind() is engine1, "FAILED: Session 2 is not bound to shared engine!"
    
    session1.close()
    session2.close()
    print("[OK] get_session() verification passed! All sessions bind to the shared singleton Engine.\n")


if __name__ == "__main__":
    test_engine_singleton()
    print("ALL ENGINE LIFECYCLE VERIFICATION TESTS PASSED SUCCESSFULLY!")
