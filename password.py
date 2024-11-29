import random
import string

def generate_password(length=12):
    # Define characters to choose from
    characters = string.ascii_letters + string.digits + string.punctuation

    # Generate password
    password = ''.join(random.choice(characters) for _ in range(length))
    return password

if __name__ == "__main__":
    # Prompt user for password length
    try:
        length = int(input("Enter the length of the password (default is 12): "))
        if length <= 0:
            raise ValueError("Length must be a positive integer")
    except ValueError:
        print("Invalid input. Using default password length of 12.")
        length = 12

    # Generate and print password
    password = generate_password(length)
    print(f"Generated Password: {password}")