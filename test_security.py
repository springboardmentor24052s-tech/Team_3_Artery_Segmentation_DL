
from dotenv import load_dotenv

load_dotenv()

from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)

password = "TestPassword123!"

hashed = hash_password(password)

print("Password hashed:", hashed != password)
print("Correct password:", verify_password(password, hashed))
print("Wrong password:", verify_password("WrongPassword", hashed))

token = create_access_token(
    user_id="test-user-123",
    role="medical_researcher",
)

payload = decode_access_token(token)

print("Token generated:", bool(token))
print("Token user ID:", payload["sub"])
print("Token role:", payload["role"])
