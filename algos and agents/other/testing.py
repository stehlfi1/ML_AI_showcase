def make_list(number):
    names = []
    for item in number:
        names.append(input("Enter name with cap"))
    print("names")

number = int(input("how many names to enter"))
names = make_list(number)
for name in names:
    if name [1] == "A":
        print("Name", name, "starts with A")