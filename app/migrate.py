import os
import sqlite3
import mysql.connector
import bcrypt
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "deepshield")

def run_migration():
    print("=== DeepShield Database Migration starting ===")
    
    # 1. Connect to MySQL without database name first to create it
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[*] Verified/Created MySQL Database: {DB_NAME}")
    except mysql.connector.Error as err:
        print(f"[ERROR] Failed to connect/create database: {err}")
        return

    # 2. Connect to MySQL with target database to create tables
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        cursor = conn.cursor()
        
        # Enable constraints
        cursor.execute("SET FOREIGN_KEY_CHECKS=0;")
        
        # Create users table
        print("[*] Creating table: users")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                first_name VARCHAR(100) NOT NULL,
                last_name VARCHAR(100) NOT NULL,
                fullname VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                phone_number VARCHAR(50),
                password_hash VARCHAR(255) NOT NULL,
                email_verified TINYINT(1) DEFAULT 0 NOT NULL,
                disabled TINYINT(1) DEFAULT 0 NOT NULL,
                is_admin TINYINT(1) DEFAULT 0 NOT NULL,
                login_attempts INT DEFAULT 0 NOT NULL,
                locked_until DATETIME NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_email (email),
                INDEX idx_is_admin (is_admin)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        
        # Create otp_verifications table
        print("[*] Creating table: otp_verifications")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS otp_verifications (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                otp_code VARCHAR(6) NOT NULL,
                expires_at DATETIME NOT NULL,
                used TINYINT(1) DEFAULT 0 NOT NULL,
                attempts INT DEFAULT 0 NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_otp (user_id, otp_code),
                INDEX idx_expires (expires_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        
        # Create password_resets table
        print("[*] Creating table: password_resets")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                reset_otp VARCHAR(6) NOT NULL,
                expires_at DATETIME NOT NULL,
                used TINYINT(1) DEFAULT 0 NOT NULL,
                attempts INT DEFAULT 0 NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_reset (user_id, reset_otp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        
        # Create user_sessions table
        print("[*] Creating table: user_sessions")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                token VARCHAR(500) NOT NULL,
                ip_address VARCHAR(45),
                device_info VARCHAR(255),
                expires_at DATETIME NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_token (token(255))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        
        # Create login_history table
        print("[*] Creating table: login_history")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS login_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NULL,
                email VARCHAR(255) NOT NULL,
                ip_address VARCHAR(45),
                device_info VARCHAR(255),
                status VARCHAR(50) NOT NULL,
                reason VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        
        # Create security_logs table
        print("[*] Creating table: security_logs")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NULL,
                email VARCHAR(255) NULL,
                action VARCHAR(100) NOT NULL,
                ip_address VARCHAR(45),
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        
        cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
        conn.commit()
        print("[*] All MySQL tables verified/created successfully.")
        
        # 3. Create default administrator account if not exists
        cursor.execute("SELECT id FROM users WHERE email = %s", ("admin@deepshield.ai",))
        admin_row = cursor.fetchone()
        if not admin_row:
            # Hash default password with bcrypt
            plain_pass = "AdminPassword123!"
            hashed_pass = bcrypt.hashpw(plain_pass.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            cursor.execute(
                "INSERT INTO users (first_name, last_name, fullname, email, password_hash, email_verified, is_admin) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                ("System", "Admin", "System Admin", "admin@deepshield.ai", hashed_pass, 1, 1)
            )
            conn.commit()
            print(f"[+] Created default Admin account: admin@deepshield.ai (Password: {plain_pass})")
        else:
            print("[*] Admin account admin@deepshield.ai already exists.")
            
        # 4. Migrate users from SQLite deepshield.db if exists
        sqlite_db = "deepshield.db"
        if os.path.exists(sqlite_db):
            print(f"[*] SQLite database '{sqlite_db}' found. Starting user migration...")
            try:
                sq_conn = sqlite3.connect(sqlite_db)
                sq_cursor = sq_conn.cursor()
                sq_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
                if sq_cursor.fetchone():
                    sq_cursor.execute("SELECT email, password, first_name, last_name, phone_number FROM users")
                    sqlite_users = sq_cursor.fetchall()
                    
                    migrated_count = 0
                    for row in sqlite_users:
                        email, password_hash, first_name, last_name, phone_number = row
                        
                        # Check if user already in MySQL
                        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                        if cursor.fetchone():
                            continue
                            
                        # Format missing columns
                        fn = first_name or "N/A"
                        ln = last_name or "N/A"
                        fullname = f"{fn} {ln}".strip()
                        if fullname == "N/A N/A" or not fullname:
                            fullname = email.split('@')[0]
                            
                        # Insert migrated user (verified = 1 by default so existing users are immediately active)
                        cursor.execute(
                            "INSERT INTO users (first_name, last_name, fullname, email, phone_number, password_hash, email_verified) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            (fn, ln, fullname, email, phone_number, password_hash, 1)
                        )
                        migrated_count += 1
                        
                    conn.commit()
                    print(f"[+] Successfully migrated {migrated_count} users from SQLite to MySQL.")
                else:
                    print("[*] No 'users' table in SQLite database to migrate.")
                sq_conn.close()
            except sqlite3.Error as sq_err:
                print(f"[ERROR] Failed to read SQLite database: {sq_err}")
        else:
            print("[*] SQLite database file not found. No users migrated.")
            
        cursor.close()
        conn.close()
        print("=== Database Migration Completed Successfully ===")
        
    except mysql.connector.Error as err:
        print(f"[ERROR] MySQL error during migration: {err}")

if __name__ == "__main__":
    run_migration()
