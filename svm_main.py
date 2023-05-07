import pandas as pd
import numpy as np
from sklearn import svm


# Convert raw data into numerical form.
def cvt_raw_data(sample):
    age = sample['age'] / 80
    is_smoke = 0 if sample['smoker'] == 'no' else 1
    is_male = 0 if sample['sex'] == 'female' else 1
    charge_value = sample['charges'] / 50000
    region_num = [0, 0, 0, 0]
    if sample['region'] == 'southeast':
        region_num[0] = 1
    elif sample['region'] == 'northeast':
        region_num[1] = 1
    elif sample['region'] == 'northwest':
        region_num[2] = 1
    else:
        region_num[3] = 1
    bmi_value = sample['bmi'] / 30
    children_num = sample['children'] / 4
    total_features = [age, is_smoke, is_male, bmi_value, children_num] + region_num
    predict_target = charge_value
    return np.array([total_features]), np.array([predict_target])


# Main entry for training the model.
def train_model():
    X = []
    Y = []
    for i in train_set['age'].keys():
        sample = {kk: train_set[kk][i] for kk in train_set.keys()}
        xx, yy = cvt_raw_data(sample)
        X.append(xx)
        Y.append(yy)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    model = svm.SVR()
    model.fit(X, Y)
    return model


def eval(model, dataset):
    total_mean = np.mean(dataset['charges'] / 50000)
    total_var = np.sum((dataset['charges'] / 50000 - total_mean) ** 2)

    total_preds = []

    for i in dataset['age'].keys():
        sample = {kk: dataset[kk][i] for kk in dataset.keys()}
        xx, yy = cvt_raw_data(sample)
        total_preds.append(model.predict(xx)[0])

    total_preds = np.array(total_preds)
    total_loss = np.sum((total_preds - dataset['charges'] / 50000) ** 2)
    print("Evaluation finished, total R square:  " + str(1 - total_loss / total_var))
    return 1 - total_loss / total_var


if __name__ == "__main__":
    k_fold = 10
    total_data_set = pd.read_csv('./data/total_data.csv')
    slice_len = total_data_set.shape[0] // k_fold
    r_squared_test = []
    r_squared_train = []
    for k in range(0, k_fold):  # k fold validation
        train_set = pd.concat([total_data_set[: k*slice_len], total_data_set[(k+1)*slice_len:]])
        test_set = total_data_set[k*slice_len: (k+1)*slice_len]
        mm = train_model()
        r_squared_train.append(eval(mm, train_set))
        r_squared_test.append(eval(mm, test_set))

    print(f'[TEST] Final results for {k_fold} fold validation:'
          f' [MEAN]={np.mean(r_squared_test)}  [STD]={np.std(r_squared_test)}')
    print(f'[TRAIN] Final results for {k_fold} fold validation:'
          f' [MEAN]={np.mean(r_squared_train)}  [STD]={np.std(r_squared_train)}')
    print("[TEST]{:.3f}±{:.3f}".format(np.mean(r_squared_test), np.std(r_squared_test)))
    print("[TRAIN]{:.3f}±{:.3f}".format(np.mean(r_squared_train), np.std(r_squared_train)))
