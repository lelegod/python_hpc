def squarecubes(arr):
    return (list(map(lambda n: n**2, arr)), list(map(lambda n: n**3, arr)))

if __name__ == "__main__":
    print(f"Output: {squarecubes([1, 2, 3, 4])}")