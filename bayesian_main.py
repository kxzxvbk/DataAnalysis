import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

bmi_range = [18, 20.5, 30, 43]    # BMI 的离散化策略
age_range = range(18, 66, 10)               # 年龄的离散化策略

# Line to partition blocks.
line1 = lambda x: 2200 + x * (23000 / 80)
line2 = lambda x: 24000 + x * (20000 / 80)
line3 = lambda x: 990000 + x * (20000 / 80)

model1 = linear_model.LinearRegression(fit_intercept=True)
model2 = linear_model.LinearRegression(fit_intercept=True)
model3 = linear_model.LinearRegression(fit_intercept=True)
model_l2 = linear_model.LinearRegression(fit_intercept=True)

y_1, age_1 = [], []     # charges for block 1, corresponding age for block 1
y_2, age_2 = [], []     # charges for block 2, corresponding age for block 2
y_3, age_3 = [], []     # charges for block 3, corresponding age for block 3

no_smoke_p = np.array([0, 0, 0], dtype=float)  # This array means: [P(no_smoke|Y=0), P(no_smoke|Y=1), P(no_smoke|Y=2)]
smoke_p = np.array([0, 0, 0], dtype=float)  # This array means: [P(smoke|Y=0), P(smoke|Y=1), P(smoke|Y=2)]

male_p = np.array([0, 0, 0], dtype=float)
female_p = np.array([0, 0, 0], dtype=float)

region_p = np.array([
    [0, 0, 0],    # southeast
    [0, 0, 0],    # northeast
    [0, 0, 0],    # northwest
    [0, 0, 0]     # southwest
], dtype=float)

bmi_p = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
], dtype=float)

children_p = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
], dtype=float)

age_p = np.zeros([len(age_range), 3], dtype=float)


# Main entry for visualizing block split.
def visualize_block_split():
    plt.scatter(train_set['age'], train_set['charges'])
    plt.plot([0, 80], [line1(0), line1(80)])
    plt.plot([0, 80], [line2(0), line2(80)])
    # plt.plot([0, 80], [line3(0), line3(80)])
    plt.title('age')
    plt.show()


def train_l2_linear_regression():
    X = []
    Y = []
    a1 = np.array(age_1).reshape(-1, 1)
    res = np.concatenate([model1.predict(a1), model2.predict(a1), model3.predict(a1)], axis=-1)
    indc = np.repeat([[1, 0, 0]], res.shape[0], axis=0)
    res = np.concatenate([res, indc], axis=-1)
    assert res.shape == (len(age_1), 6)
    X.append(res)
    Y.append(np.array(y_1).reshape(-1, 1))

    a2 = np.array(age_2).reshape(-1, 1)
    res = np.concatenate([model1.predict(a2), model2.predict(a2), model3.predict(a2)], axis=-1)
    indc = np.repeat([[0, 1, 0]], res.shape[0], axis=0)
    res = np.concatenate([res, indc], axis=-1)
    assert res.shape == (len(age_2), 6)
    X.append(res)
    Y.append(np.array(y_2).reshape(-1, 1))

    a3 = np.array(age_3).reshape(-1, 1)
    res = np.concatenate([model1.predict(a3), model2.predict(a3), model3.predict(a3)], axis=-1)
    indc = np.repeat([[0, 0, 1]], res.shape[0], axis=0)
    res = np.concatenate([res, indc], axis=-1)
    assert res.shape == (len(age_3), 6)
    X.append(res)
    Y.append(np.array(y_3).reshape(-1, 1))

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    model_l2.fit(X, Y)
    print(f"k for l2 model: {model_l2.coef_}    b for l2 model: {model_l2.intercept_}")


# Main entry for training Linear Regressor.
def train_linear_regression():
    # Block 1
    a1 = np.array(age_1).reshape(-1, 1)
    y1 = np.array(y_1).reshape(-1, 1)
    model1.fit(a1, y1)
    print(f"k for Block 1: {model1.coef_}    b for Block 1: {model1.intercept_}")

    # Block 2
    a2 = np.array(age_2).reshape(-1, 1)
    y2 = np.array(y_2).reshape(-1, 1)
    model2.fit(a2, y2)
    print(f"k for Block 2: {model2.coef_}    b for Block 2: {model2.intercept_}")

    # Block 3
    a3 = np.array(age_3).reshape(-1, 1)
    y3 = np.array(y_3).reshape(-1, 1)
    model3.fit(a3, y3)
    print(f"k for Block 3: {model3.coef_}    b for Block 1: {model3.intercept_}")

    return model1, model2, model3


# Main entry for training Bayesian classifier.
def train_bayesian():
    global smoke_p
    global no_smoke_p
    global female_p
    global male_p
    global age_p
    global children_p
    global bmi_p
    global region_p

    avg0, num0 = 0, 0  # The averaged value and sample number of block 0.
    avg1, num1 = 0, 0  # The averaged value and sample number of block 1.
    avg2, num2 = 0, 0  # The averaged value and sample number of block 2.

    for i in train_set['age'].keys():
        # Transform original data into numerical & discrete form.
        age = train_set['age'][i]

        is_smoke = False if train_set['smoker'][i] == 'no' else True
        is_male = False if train_set['sex'][i] == 'female' else True
        charge_value = train_set['charges'][i]
        if train_set['region'][i] == 'southeast':
            region_num = 0
        elif train_set['region'][i] == 'northeast':
            region_num = 1
        elif train_set['region'][i] == 'northwest':
            region_num = 2
        else:
            region_num = 3
        # Dealing with BMI.
        bmi_num = -1
        bmi_value = train_set['bmi'][i]
        for ind in range(len(bmi_range)):
            if bmi_range[ind] > bmi_value:
                bmi_num = ind
                break
        if bmi_num == -1:
            bmi_num = len(bmi_range)
        children_num = train_set['children'][i]

        # Label this sample as 0 or 1 or 2.
        split1 = line1(age)
        split2 = line2(age)
        split3 = line3(age)

        if charge_value < split1:  # Class 0
            block_num = 0
            avg0 += charge_value
            num0 += 1
            y_1.append(charge_value / 50000)
            age_1.append(age)
        elif split3 > charge_value > split2:  # Class 2
            block_num = 2
            avg2 += charge_value
            num2 += 1
            y_3.append(charge_value / 50000)
            age_3.append(age)
        elif split2 > charge_value > split1:  # Class 1
            block_num = 1
            avg1 += charge_value
            num1 += 1
            y_2.append(charge_value / 50000)
            age_2.append(age)
        else:
            continue

        # Update P(X|Y)
        if is_smoke:
            smoke_p[block_num] += 1
        else:
            no_smoke_p[block_num] += 1

        if is_male:
            male_p[block_num] += 1
        else:
            female_p[block_num] += 1

        age_num = train_set['age'][i]
        age_index = -1
        for idd in range(len(age_range) - 1):
            if age_range[idd] <= age_num < age_range[idd+1]:
                age_index = idd
                break
        age_index = age_index if age_index > 0 else len(age_range) - 1
        age_p[age_index][block_num] += 1
        region_p[region_num][block_num] += 1
        bmi_p[bmi_num][block_num] += 1
        children_p[children_num][block_num] += 1

    tmp_region = []
    for r in region_p:
        tmp_region.append(r / np.sum(r))
    print(tmp_region)
    assert False


    # Normalize each P(X|Y) into [0, 1]
    pre_prob = np.array([num0, num1, num2])
    smoke_p /= pre_prob
    no_smoke_p /= pre_prob
    male_p /= pre_prob
    female_p /= pre_prob
    region_p /= pre_prob
    bmi_p /= pre_prob
    children_p /= pre_prob
    age_p /= pre_prob

    print(f"Mean charge for block0 {avg0 / num0}")
    print(f"Mean charge for block1 {avg1 / num1}")
    print(f"Mean charge for block2 {avg2 / num2}")


def evaluate(is_male, is_smoker, region_num, bmi_num, children_num, age, keys, l2_enabled=False):
    p = np.array([1., 1., 1.])

    if 'sex' in keys:
        if is_male:
            p *= male_p
        else:
            p *= female_p

    if 'smoker' in keys:
        if is_smoker:
            p *= smoke_p
        else:
            p *= no_smoke_p

    if 'region' in keys:
        p *= region_p[region_num]

    if 'bmi' in keys:
        p *= bmi_p[bmi_num]

    if 'children' in keys:
        p *= children_p[children_num]

    if 'age' in keys:
        age_index = -1
        for idd in range(len(age_range) - 1):
            if age_range[idd] <= age < age_range[idd + 1]:
                age_index = idd
                break
        age_index = age_index if age_index > 0 else len(age_range) - 1
        p *= age_p[age_index]

    index = np.argmax(p)
    models = [model1, model2, model3]
    results = [models[ii].predict([[age]])[0][0] for ii in range(3)]
    indicator = [0, 0, 0]
    indicator[index] = 1
    if not l2_enabled:
        return models[index].predict([[age]])[0][0]
    else:
        return model_l2.predict([results + indicator])[0][0]


def eval(dataset, keys):
    total_mean = np.mean(dataset['charges'] / 50000)
    total_var = np.sum((dataset['charges'] / 50000 - total_mean) ** 2)

    total_preds = []

    for i in dataset['age'].keys():
        age = dataset['age'][i]
        is_smoke = False if dataset['smoker'][i] == 'no' else True
        is_male = False if dataset['sex'][i] == 'female' else True

        if dataset['region'][i] == 'southeast':
            region_num = 0
        elif dataset['region'][i] == 'northeast':
            region_num = 1
        elif dataset['region'][i] == 'northwest':
            region_num = 2
        else:
            region_num = 3

        bmi_num = -1
        bmi_value = dataset['bmi'][i]
        for ind in range(len(bmi_range)):
            if bmi_range[ind] > bmi_value:
                bmi_num = ind
                break
        if bmi_num == -1:
            bmi_num = len(bmi_range)

        children_num = dataset['children'][i]
        total_preds.append(evaluate(is_male, is_smoke, region_num, bmi_num, children_num, age, keys))

    total_preds = np.array(total_preds)
    total_loss = np.sum((total_preds - dataset['charges'] / 50000) ** 2)
    print("Evaluation finished, total R square:  " + str(1 - total_loss / total_var))
    return 1 - total_loss / total_var


if __name__ == "__main__":
    k_fold = 10
    total_data_set = pd.read_csv('./data/total_data.csv')

    bayesian_keys = ['smoker', 'bmi']
    slice_len = total_data_set.shape[0] // k_fold
    r_squared_test = []
    r_squared_train = []
    for k in range(0, k_fold):  # k fold validation
        train_set = pd.concat([total_data_set[: k*slice_len], total_data_set[(k+1)*slice_len:]])
        test_set = total_data_set[k*slice_len: (k+1)*slice_len]
        # visualize_block_split()
        train_bayesian()
        train_linear_regression()
        train_l2_linear_regression()
        r_squared_train.append(eval(train_set, bayesian_keys))
        r_squared_test.append(eval(test_set, bayesian_keys))

    print(f'[TEST] Final results for {k_fold} fold validation:'
          f' [MEAN]={np.mean(r_squared_test)}  [STD]={np.std(r_squared_test)}')
    print(f'[TRAIN] Final results for {k_fold} fold validation:'
          f' [MEAN]={np.mean(r_squared_train)}  [STD]={np.std(r_squared_train)}')
    print("[TEST]{:.3f}±{:.3f}".format(np.mean(r_squared_test), np.std(r_squared_test)))
    print("[TRAIN]{:.3f}±{:.3f}".format(np.mean(r_squared_train), np.std(r_squared_train)))
