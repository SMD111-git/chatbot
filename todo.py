# Function to display the todo list
def display_todo_list(todo_list):
    if not todo_list:
        print("Todo list is empty")
    else:
        print("Todo List:")
        for index, item in enumerate(todo_list, start=1):
            print(f"{index}. {item}")

# Function to add an item to the todo list
def add_todo_item(todo_list, item):
    todo_list.append(item)
    print(f"Added '{item}' to the todo list.")

# Function to remove an item from the todo list
def remove_todo_item(todo_list, index):
    if 1 <= index <= len(todo_list):
        item = todo_list.pop(index - 1)
        print(f"Removed '{item}' from the todo list.")
    else:
        print("Invalid index. Please try again.")

# Main program
def main():
    todo_list = []

    while True:
        print("\nTodo List Menu:")
        print("1. Display Todo List")
        print("2. Add Item to Todo List")
        print("3. Remove Item from Todo List")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            display_todo_list(todo_list)
        elif choice == '2':
            item = input("Enter the item to add: ")
            add_todo_item(todo_list, item)
        elif choice == '3':
            index = int(input("Enter the index of item to remove: "))
            remove_todo_item(todo_list, index)
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 4.")

if __name__ == "__main__":
    main()