import unittest
import sys

def main():
    print("=" * 60)
    print(" Running RAG Pipeline Test Suite")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    if result.wasSuccessful():
        print(f"[PASS] All {result.testsRun} tests passed successfully!")
        sys.exit(0)
    else:
        print(f"[FAIL] {len(result.failures)} failure(s), {len(result.errors)} error(s).")
        sys.exit(1)

if __name__ == "__main__":
    main()

