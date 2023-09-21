# x = input().split()

# a = int(x[0])
# b = int(x[1])
# print(a + b)
# print(a - b)
# print(a * b)
# print(a // b)
# print(a / b)

# x = input()
# print(int(x[:3]))
# print(int(x[3:6]))
# print(int(x[6:9]))

# x = input()
# y = x[::-1]
# print(int(y))

# x = float(input())
# y = x ** 2 + 2 * x - 10
# y -= int(y)
# print(str(y).split('.')[1])

# x = input().split()

# mass = float(x[0])
# orig_t = float(x[1])
# final_t = float(x[2])

# # Q=M*(最终温度-初始温度)*4184
# energy = mass * (final_t - orig_t) * 4184
# print(energy)

mins = int(input())

mins_out = mins % 60
hours = mins // 60

hours_out = hours % 24
days = hours // 24

days_out = days % 365
years = days // 365

years_out = years
print('{}years {}days {}hours {}mins'.format(years_out, days_out, hours_out, mins_out))