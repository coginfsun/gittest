class MyClass:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"MyClass object with name: {self.name}"

# Create an instance of MyClass
obj = MyClass("example")

# Print the object, which will call the __str__ magic function
print(obj)