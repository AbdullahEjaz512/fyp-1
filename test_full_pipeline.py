"""
Complete End-to-End Test for Ensemble Integration
Tests backend API, frontend connectivity, and uncertainty display
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")

def test_backend_health():
    """Test if backend server is running"""
    print_header("TEST 1: Backend Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("Backend server is running")
            data = response.json()
            print_info(f"Service: {data.get('service', 'Unknown')}")
            return True
        else:
            print_error(f"Backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Backend server is not running!")
        print_info("Start with: cd backend && python -m uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print_error(f"Backend health check failed: {e}")
        return False

def test_ensemble_status():
    """Test ensemble models status"""
    print_header("TEST 2: Ensemble Models Status")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/ensemble/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Overall status
            if data['ensemble_enabled'] and data['ensemble_available']:
                print_success("Ensemble models are ENABLED and AVAILABLE")
            else:
                print_warning("Ensemble models are NOT fully available")
                return False
            
            # Segmentation
            seg = data['models']['segmentation']
            print(f"\n{Colors.BOLD}Segmentation Ensemble:{Colors.RESET}")
            print(f"  • Initialized: {Colors.GREEN if seg['ensemble_initialized'] else Colors.RED}{seg['ensemble_initialized']}{Colors.RESET}")
            print(f"  • Models: {seg['num_models']}")
            print(f"  • Features: {', '.join(seg['features'])}")
            
            # Classification
            cls = data['models']['classification']
            print(f"\n{Colors.BOLD}Classification Ensemble:{Colors.RESET}")
            print(f"  • Initialized: {Colors.GREEN if cls['ensemble_initialized'] else Colors.RED}{cls['ensemble_initialized']}{Colors.RESET}")
            print(f"  • Models: {cls['num_models']}")
            print(f"  • Method: {cls['method']}")
            print(f"  • Features: {', '.join(cls['features'])}")
            
            # Expected improvements
            print(f"\n{Colors.BOLD}Expected Improvements:{Colors.RESET}")
            for key, value in data['expected_improvements'].items():
                print(f"  • {key}: {Colors.GREEN}{value}{Colors.RESET}")
            
            return True
        else:
            print_error(f"Ensemble status endpoint returned {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Ensemble status check failed: {e}")
        return False

def test_frontend_running():
    """Test if frontend dev server is running"""
    print_header("TEST 3: Frontend Server Check")
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200:
            print_success("Frontend dev server is running")
            print_info(f"URL: {FRONTEND_URL}")
            return True
        else:
            print_warning(f"Frontend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Frontend dev server is not running!")
        print_info("Start with: cd frontend && npm run dev")
        return False
    except Exception as e:
        print_error(f"Frontend check failed: {e}")
        return False

def check_auth_token():
    """Check if user has auth token"""
    print_header("TEST 4: Authentication Check")
    
    print_info("Checking for saved authentication token...")
    print_warning("This test requires manual verification")
    print()
    print("To test the full pipeline, you need to:")
    print("  1. Open browser: http://localhost:5173")
    print("  2. Login with your credentials")
    print("  3. Check browser console for authToken in localStorage")
    print()
    
    response = input(f"{Colors.YELLOW}Are you logged in? (y/n): {Colors.RESET}")
    return response.lower() == 'y'

def print_manual_test_steps():
    """Print manual testing instructions"""
    print_header("MANUAL TESTING STEPS")
    
    steps = [
        ("Open Browser", "Navigate to http://localhost:5173"),
        ("Login", "Use your doctor/patient credentials"),
        ("Go to Upload Page", "Click 'Upload' in navigation"),
        ("Upload MRI Scan", "Select a .nii or .nii.gz file (BraTS format preferred)"),
        ("Wait for Upload", "Progress bar should show 100%"),
        ("Click 'Analyze'", "This triggers the ensemble prediction"),
        ("Wait for Analysis", "Should take 2-5 seconds"),
        ("View Results", "Click 'View Results' when analysis completes"),
        ("Check Uncertainty Card", "Look for 'Ensemble AI - Uncertainty Analysis' card"),
        ("Verify Display", "Check confidence bars, quality badges, and metrics")
    ]
    
    print(f"{Colors.BOLD}Follow these steps to test the full pipeline:{Colors.RESET}\n")
    for i, (step, description) in enumerate(steps, 1):
        print(f"{Colors.CYAN}{i:2d}.{Colors.RESET} {Colors.BOLD}{step}{Colors.RESET}")
        print(f"     {description}")
        print()

def print_what_to_look_for():
    """Print what users should see in the UI"""
    print_header("WHAT TO LOOK FOR IN THE UI")
    
    print(f"{Colors.BOLD}1. Ensemble Uncertainty Card:{Colors.RESET}")
    print("   • Should appear between summary and segmentation sections")
    print("   • Has cyan gradient background with shield icon")
    print("   • Title: 'Ensemble AI - Uncertainty Analysis'")
    print()
    
    print(f"{Colors.BOLD}2. Segmentation Quality Panel:{Colors.RESET}")
    print("   • Confidence score with progress bar (e.g., '87.3%')")
    print("   • Uncertainty/entropy score (e.g., '23.1%')")
    print("   • Quality badges:")
    print("     - ✓ High Confidence (green)")
    print("     - ✓ Low Uncertainty (green)")
    print("     - ✓ Clinical Ready (cyan)")
    print("     - ⚠ Expert Review Needed (yellow, if applicable)")
    print()
    
    print(f"{Colors.BOLD}3. Classification Quality Panel:{Colors.RESET}")
    print("   • Model uncertainty score (e.g., '15.2%')")
    print("   • Same quality badges as segmentation")
    print()
    
    print(f"{Colors.BOLD}4. Color Coding:{Colors.RESET}")
    print("   • Green bars = High confidence (> 80%)")
    print("   • Yellow bars = Medium confidence (60-80%)")
    print("   • Red bars = Low confidence (< 60%)")
    print()
    
    print(f"{Colors.BOLD}5. Information Footer:{Colors.RESET}")
    print("   • Explains ensemble technology")
    print("   • Shows expected improvements (+3-5% segmentation, +2-4% classification)")

def print_troubleshooting():
    """Print troubleshooting guide"""
    print_header("TROUBLESHOOTING")
    
    issues = [
        ("Ensemble card not showing", [
            "Check browser console for errors",
            "Verify backend returned 'ensemble' field in response",
            "Try hard refresh (Ctrl+Shift+R)",
            "Check Network tab for /api/v1/analyze response"
        ]),
        ("Backend 500 error during analysis", [
            "Check backend console logs",
            "Verify model files exist in ml_models/",
            "Restart backend server",
            "Check GPU/CPU memory availability"
        ]),
        ("Frontend not updating", [
            "Check if dev server auto-reloaded",
            "Manually restart: npm run dev",
            "Clear browser cache",
            "Check for TypeScript errors in terminal"
        ]),
        ("Confidence bars not showing", [
            "Verify ensemble.segmentation_uncertainty exists in API response",
            "Check browser console for render errors",
            "Inspect element to see if styles applied correctly"
        ])
    ]
    
    for issue, solutions in issues:
        print(f"{Colors.YELLOW}Issue: {issue}{Colors.RESET}")
        for solution in solutions:
            print(f"  • {solution}")
        print()

def main():
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║         ENSEMBLE AI - END-TO-END INTEGRATION TEST                ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")
    
    print_info(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run automated tests
    results = {
        'backend': test_backend_health(),
        'ensemble': False,
        'frontend': False,
        'auth': False
    }
    
    if results['backend']:
        results['ensemble'] = test_ensemble_status()
    
    results['frontend'] = test_frontend_running()
    
    if results['backend'] and results['ensemble'] and results['frontend']:
        results['auth'] = check_auth_token()
    
    # Summary
    print_header("TEST SUMMARY")
    
    all_passed = all(results.values())
    
    print(f"Backend Server:     {Colors.GREEN + '✓ PASS' if results['backend'] else Colors.RED + '✗ FAIL'}{Colors.RESET}")
    print(f"Ensemble Models:    {Colors.GREEN + '✓ PASS' if results['ensemble'] else Colors.RED + '✗ FAIL'}{Colors.RESET}")
    print(f"Frontend Server:    {Colors.GREEN + '✓ PASS' if results['frontend'] else Colors.RED + '✗ FAIL'}{Colors.RESET}")
    print(f"Authentication:     {Colors.GREEN + '✓ READY' if results['auth'] else Colors.YELLOW + '⚠ PENDING'}{Colors.RESET}")
    
    print()
    
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL SYSTEMS READY!{Colors.RESET}")
        print()
        print_manual_test_steps()
        print_what_to_look_for()
    else:
        print(f"{Colors.RED}{Colors.BOLD}⚠ SOME TESTS FAILED{Colors.RESET}")
        print()
        if not results['backend']:
            print_error("Start backend: cd backend && python -m uvicorn app.main:app --reload")
        if not results['frontend']:
            print_error("Start frontend: cd frontend && npm run dev")
    
    print_troubleshooting()
    
    print_header("NEXT STEPS")
    print("1. Complete the manual testing steps above")
    print("2. Upload a brain MRI scan (BraTS format recommended)")
    print("3. Run analysis and verify ensemble uncertainty display")
    print("4. Take screenshots of the uncertainty card")
    print("5. Report any issues or unexpected behavior")
    print()
    print_info("For support, check logs in backend/logs/ and browser console")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n\n{Colors.RED}Unexpected error: {e}{Colors.RESET}")
