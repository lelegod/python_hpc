import sys

def grades_check():
    grades = [int(g) for g in sys.argv[1:]]
    average = sum(grades) / len(grades)
    return f"{average} {'Pass' if average >= 5 else 'Fail'}"

if __name__ == "__main__":
    print(f"{grades_check()}")