import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Plot total data as histogram.
def plot_data(charges, box_num, name):
    a = charges
    a_min = int(np.min(a))
    a_max = int(np.max(a))
    bin_len = (a_max - a_min) // box_num
    plt.hist(a, bins=range(a_min, a_max+1, bin_len))
    plt.title(name)
    plt.show()


# Plot discrete variables.
def div_by(class_name, options):
    class1 = []
    class2 = []
    for i in range(train_set.shape[0]):
        if train_set[class_name][i] == options[0]:
            class1.append(train_set['charges'][i])
        else:
            class2.append(train_set['charges'][i])
    return class1, class2


# Plot total data
train_set = pd.read_csv('./data/total_data.csv')
plot_data(train_set['charges'], 100, 'total')

# Plot BMI & charge (also partition for bins)
plt.scatter(train_set['bmi'], train_set['charges'])
plt.plot([18, 18], [0, 80000])
plt.plot([20.5, 20.5], [0, 80000])
plt.plot([30, 30], [0, 80000])
plt.plot([43, 43], [0, 80000])
plt.title('bmi')
plt.show()

# Plot children & charge
plt.scatter(train_set['children'], train_set['charges'])
plt.title('children')
plt.show()

# Plot age & charge (also regression results)
plt.scatter(train_set['age'], train_set['charges'])
k = [268.38184306, 265.10207016, 278.05950774]
b = [-3492.51400728, 11402.2185401, 29993.27304893]
plt.plot([0, 80], [b[0], 80 * k[0] + b[0]])
plt.plot([0, 80], [b[1], 80 * k[1] + b[1]])
plt.plot([0, 80], [b[2], 80 * k[2] + b[2]])
plt.title('age')
plt.show()

# Plot region & charge
plt.scatter(train_set['region'], train_set['charges'])
plt.title('region')
plt.show()

# Plot smoke & charge
no_smoke, smoke = div_by('smoker', ['no', 'yes'])
plot_data(no_smoke, 100, 'no smoke')
plot_data(smoke, 100, 'smoke')

# Plot sex & charge
male, female = div_by('sex', ['male', 'female'])
plot_data(male, 100, 'male')
plot_data(female, 100, 'female')
