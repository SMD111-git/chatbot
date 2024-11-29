import random

def guess_number_game():
    print("Welcome to the Number Guessing Game!")
    print("I'm thinking of a number between 1 and 100.")

    # Generate a random number between 1 and 100
    secret_number = random.randint(1, 100)
    attempts = 0
    max_attempts = 10

    while attempts < max_attempts:
        print("\nAttempts left:", max_attempts - attempts)
        guess = input("Enter your guess (between 1 and 100): ")

        # Validate user input
        if not guess.isdigit():
            print("Please enter a valid number.")
            continue

        guess = int(guess)
        attempts += 1

        # Check if the guess is correct, too high, or too low
        if guess < secret_number:
            print("Too low. Try again.")
        elif guess > secret_number:
            print("Too high. Try again.")
        else:
            print(f"Congratulations! You guessed the number {secret_number} in {attempts} attempts.")
            break
    else:
        print(f"\nGame over! You did not guess the number. The correct number was {secret_number}.")

if __name__ == "__main__":
    guess_number_game()