import sys

def even_only():
    nums = [int(g) for g in sys.argv[1:]]
    return list(filter(lambda n: n % 2 == 0, nums))

if __name__ == "__main__":
    print(f"{even_only()}")