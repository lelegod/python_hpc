def sorttuples(arr):
    return sorted(arr, key=lambda t: t[1])

if __name__ == "__main__":
    print(f"Output: {sorttuples([(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)])}")