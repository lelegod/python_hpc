def deduplicate(arr):
    return list(set(arr))

if __name__ == "__main__":
    print(f"Output: {deduplicate([1, 1, 2, 3.3])}")