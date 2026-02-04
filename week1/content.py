def main(text: str = "Hello world"):
    print(text)
    with open("content.txt", "w") as f:
        f.write(text)

if __name__ == "__main__":
    main()