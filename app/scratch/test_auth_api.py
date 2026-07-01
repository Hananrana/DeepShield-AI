import requests
import json
import sys
import os
import re

# Add parent directory to path to import db module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db

BASE_URL = "http://127.0.0.1:5000"

def get_csrf_token(session, path="/login"):
    """Fetches a page and extracts the CSRF token from the meta tag."""
    url = f"{BASE_URL}{path}"
    resp = session.get(url)
    if resp.status_code != 200:
        print(f"[FAIL] Could not load GET {path} to obtain CSRF token. Status: {resp.status_code}")
        return None
        
    # Simple regex search for meta content
    match = re.search(r'<meta name="csrf-token" content="([^"]+)">', resp.text)
    if not match:
        print(f"[FAIL] CSRF token meta tag not found in {path}")
        return None
        
    token = match.group(1)
    session.headers.update({"X-CSRF-Token": token})
    return token

def get_debug_otp():
    """Reads the generated OTP from static/uploads/otp_debug.json."""
    debug_path = "static/uploads/otp_debug.json"
    if not os.path.exists(debug_path):
        print("[FAIL] otp_debug.json not found on disk.")
        return None
    try:
        with open(debug_path, "r") as f:
            data = json.load(f)
            return data.get("otp")
    except Exception as e:
        print(f"[FAIL] Failed to read otp_debug.json: {e}")
        return None

def test_flow():
    # Clean up integration test user first for complete test idempotency
    email = "integration_test_user@deepshield.ai"
    print(f"Cleaning up legacy records for {email}...")
    try:
        db.execute_query("DELETE FROM users WHERE email = %s", (email,), commit=True)
        print("[PASS] Cleaned up test user records.")
    except Exception as db_err:
        print(f"[WARN] Failed to delete test user: {db_err}")

    session = requests.Session()
    print("=== Starting Upgraded Authentication Integration Test ===")

    # 1. Fetch CSRF token
    csrf = get_csrf_token(session, "/signup")
    if not csrf:
        return False
    print(f"[PASS] Acquired CSRF Token: {csrf[:10]}...")

    # 2. Test Signup API
    signup_payload = {
        "first_name": "Integration",
        "last_name": "Tester",
        "email": email,
        "phone_number": "+1555123456",
        "password": "SecurePassword123!",
        "confirm_password": "SecurePassword123!"
    }
    
    print("\n2. Testing Signup API (expecting inactive user + OTP trigger)...")
    resp = session.post(f"{BASE_URL}/api/signup", json=signup_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    if resp.status_code != 200:
        print("[FAIL] Signup failed.")
        return False
    print("[PASS] Signup API success.")

    # Get OTP
    otp = get_debug_otp()
    if not otp:
        print("[FAIL] Verification OTP code was not generated/found.")
        return False
    print(f"[PASS] Retrieved verification OTP: {otp}")

    # 3. Test Verify OTP API
    print("\n3. Testing OTP Verification API...")
    verify_payload = {
        "email": email,
        "otp": otp
    }
    resp = session.post(f"{BASE_URL}/api/verify-otp", json=verify_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    if resp.status_code != 200:
        print("[FAIL] OTP verification failed.")
        return False
    print("[PASS] Email successfully verified.")

    # 4. Refresh CSRF token from Login page
    csrf = get_csrf_token(session, "/login")
    print(f"[PASS] Refreshed CSRF Token: {csrf[:10]}...")

    # 5. Test Login API
    print("\n5. Testing Login API (expecting success + cookies)...")
    login_payload = {
        "email": email,
        "password": "SecurePassword123!"
    }
    resp = session.post(f"{BASE_URL}/api/login", json=login_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    print(f"Cookies Set: {session.cookies.get_dict()}")
    if resp.status_code != 200:
        print("[FAIL] Login failed.")
        return False
    if "access_token" not in session.cookies or "refresh_token" not in session.cookies:
        print("[FAIL] Access/Refresh token cookies were not set on login.")
        return False
    print("[PASS] Login API succeeded and JWT cookies are present.")

    # 6. Test Change Password (Logged in)
    print("\n6. Testing Change Password API (Current -> New)...")
    change_payload = {
        "current_password": "SecurePassword123!",
        "new_password": "NewSecurePassword456!",
        "confirm_password": "NewSecurePassword456!"
    }
    resp = session.post(f"{BASE_URL}/api/change-password", json=change_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    if resp.status_code != 200:
        print("[FAIL] Password change failed.")
        return False
    print("[PASS] Password changed successfully.")

    # 7. Test Logout
    print("\n7. Testing Logout (Clear session/cookies)...")
    resp = session.get(f"{BASE_URL}/logout")
    print(f"Status Code: {resp.status_code}")
    print(f"Cookies after Logout: {session.cookies.get_dict()}")
    if "access_token" in session.cookies and session.cookies["access_token"]:
        print("[FAIL] Cookies were not cleared on logout.")
        return False
    print("[PASS] Logout cleared sessions and cookies.")

    # 8. Try to login with OLD password (should fail)
    csrf = get_csrf_token(session, "/login")
    print("\n8. Testing Login with OLD password (expecting failure)...")
    resp = session.post(f"{BASE_URL}/api/login", json=login_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    if resp.status_code == 200:
        print("[FAIL] Logged in with outdated password.")
        return False
    print("[PASS] Outdated password rejected.")

    # 9. Login with NEW password
    print("\n9. Testing Login with NEW password (expecting success)...")
    login_payload["password"] = "NewSecurePassword456!"
    resp = session.post(f"{BASE_URL}/api/login", json=login_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    if resp.status_code != 200:
        print("[FAIL] Login with new password failed.")
        return False
    print("[PASS] Logged in successfully with new password.")

    # 10. Forgot password request (OTP flow)
    print("\n10. Testing Forgot Password OTP request...")
    forgot_payload = {
        "email": email
    }
    resp = session.post(f"{BASE_URL}/api/forgot-password", json=forgot_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    if resp.status_code != 200:
        print("[FAIL] Forgot Password API call failed.")
        return False
    
    reset_otp = get_debug_otp()
    if not reset_otp:
        print("[FAIL] Reset OTP code not generated.")
        return False
    print(f"[PASS] Acquired reset OTP code: {reset_otp}")

    # 11. Reset password via OTP
    print("\n11. Testing Reset Password via recovery OTP...")
    reset_payload = {
        "email": email,
        "otp": reset_otp,
        "password": "FinalSecurePassword789!",
        "confirm_password": "FinalSecurePassword789!"
    }
    resp = session.post(f"{BASE_URL}/api/reset-password", json=reset_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    if resp.status_code != 200:
        print("[FAIL] Password reset via OTP failed.")
        return False
    print("[PASS] Password reset successfully using recovery OTP.")

    # 12. Verify login with final password
    print("\n12. Testing Login with final password...")
    login_payload["password"] = "FinalSecurePassword789!"
    resp = session.post(f"{BASE_URL}/api/login", json=login_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    if resp.status_code != 200:
        print("[FAIL] Login with final password failed.")
        return False
    print("[PASS] Logged in successfully with final password.")

    # 13. Test unauthorized access to Admin APIs (should fail)
    print("\n13. Testing admin restrictions on current user (non-admin)...")
    resp = session.get(f"{BASE_URL}/api/admin/logs")
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    if resp.status_code == 200:
        print("[FAIL] Non-admin user obtained access to admin logs.")
        return False
    elif resp.status_code == 403:
        print("[PASS] Access denied correctly (403 Forbidden).")
    else:
        print(f"[WARN] Unexpected status code: {resp.status_code}")

    print("\n================================================")
    print("[SUCCESS] All programmatic integration tests passed!")
    print("================================================")
    return True

if __name__ == "__main__":
    success = test_flow()
    sys.exit(0 if success else 1)
