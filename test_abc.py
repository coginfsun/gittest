def test_map():
    # Test case 1: Mapping a function to a list of integers
    result = list(map(lambda x: x * 2, [1, 2, 3, 4]))
    assert result == [2, 4, 6, 8], "Test case 1 failed"

    # Test case 2: Mapping a function to an empty list
    result = list(map(lambda x: x.upper(), []))
    assert result == [], "Test case 2 failed"

    # Test case 3: Mapping a function to a list of strings
    result = list(map(lambda x: x.lower(), ["HELLO", "WORLD"]))
    assert result == ["hello", "world"], "Test case 3 failed"

    # Add more test cases here...

    print("All test cases passed")

test_map()