import py_compile
import sys

try:
    py_compile.compile('backend/app/routers/advanced_modules.py', doraise=True)
    print("Syntax check passed: backend/app/routers/advanced_modules.py")
except Exception as e:
    print(f"Syntax check failed: {e}")
    sys.exit(1)
