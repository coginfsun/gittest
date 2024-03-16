from functools import reduce

# Define a function to add two numbers
def add(x, y):
    return x + y

# Create a list of numbers
numbers = [1, 2, 3, 4, 5]

# Use reduce() to add all the numbers in the list
result = reduce(add, numbers)

# Print the result
print(result)