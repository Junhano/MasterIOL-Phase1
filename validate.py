#Library imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from util import random_split_ds
import matplotlib.pyplot as plt
import numpy as np

#Builtin imports
import copy
import os
from sklearn import metrics



def call_validate_func(model: type, validatate_func, train_dataset, validate_dataset, eval = True):
    train_accuracy, validate_accuracy, false_positive_rate, false_negative_rate,\
    true_positive_roc, false_positive_roc = validatate_func(model, train_dataset, validate_dataset, eval)
    return train_accuracy, validate_accuracy, false_positive_rate, false_negative_rate, true_positive_roc, false_positive_roc

def crossfold_validation(model: type, train_func, validate_func, dataset, device = 'cpu', KFold = 6, draw = False,
                         ROC=False, stratified = True, class_index=1):
    train_accuracy_list = []
    accuracy = []
    false_positive = []
    false_negative = []



    split_dataset = random_split_ds(dataset, KFold, stratified, class_index)

    try:
        train_model = model()
    except:
        train_model = model



    cross_validation_loss_tracker = []
    False_positive_ROC = [0 for _ in range(11)]
    True_positive_ROC = [0 for _ in range(11)]
    print(f'Cross Fold Validation with train_model: {train_model}')
    for i in range(KFold):
        concat_list = [split_dataset[k] for k in range(len(split_dataset)) if k != i]
        concat_dataset = ConcatDataset(concat_list)
        validation_dataset = split_dataset[i]

        temp_model = copy.deepcopy(train_model).to(device)

        stage, train_loss_tracker = train_func(temp_model, concat_dataset, call_validate_func, validate_func, validation_dataset)
        cross_validation_loss_tracker.append(train_loss_tracker)

        train_accuracy, validate_accuracy, false_positive_rate, false_negative_rate, temp_true_positive_roc, temp_false_positive_rate = \
            call_validate_func(temp_model, validate_func, concat_dataset, validation_dataset)

        if ROC:
            for p in range(11):
                False_positive_ROC[p] += (temp_false_positive_rate[p] - False_positive_ROC[p]) / (i+1)
                True_positive_ROC[p] += (temp_true_positive_roc[p] - True_positive_ROC[p]) / (i+1)

        accuracy.append(validate_accuracy)
        train_accuracy_list.append(train_accuracy)
        false_positive.append(false_positive_rate)
        false_negative.append(false_negative_rate)

    if draw:

        epoch_training_err = np.average(np.array(cross_validation_loss_tracker), axis = 0)
        index = [i + 1 for i in range(len(epoch_training_err))]

        plt.plot(index, epoch_training_err, label='Epoch Training Error')

        plt.xticks(index)
        plt.legend()
        plt.show()

    if ROC and draw:
        print(False_positive_ROC)
        print(True_positive_ROC)
        x = [0, 1]
        y = [0, 1]
        plt.plot(x, y)
        plt.plot(False_positive_ROC, True_positive_ROC, 'go-', linewidth = 2)
        plt.show()


    print(accuracy)
    if len(accuracy) != 0:
        print('Validation Accuracy for this model is {}'.format(100 * (sum(accuracy) / len(accuracy))))
        print('Training Accuracy for this model is {}'.format(100 * (sum(train_accuracy_list) / len(train_accuracy_list))))
        print('Average False Positive rate for this model is {}%'.format(100 * (sum(false_positive) / len(false_positive))))
        print('Average False Negative rate for this model is {}%'.format(100 * (sum(false_negative)/ len(false_negative))))


    AUC = 0
    if ROC:
        AUC = metrics.auc(False_positive_ROC, True_positive_ROC)

    return 100 * (sum(train_accuracy_list) / len(train_accuracy_list)), \
           100 * (sum(accuracy) / len(accuracy)), \
           100 * (sum(false_positive) / len(false_positive)), \
           100 * (sum(false_negative)/ len(false_negative)), AUC, False_positive_ROC, True_positive_ROC

def validate_stage1(saved_state_path = None, device = 'cpu', cross_validate = False):
    def validate(model, train_dataset, validate_dataset, eval):
        print('Validate using stage 1 method')
        if not cross_validate:
            test_model = model()
            if saved_state_path != None and os.path.exists(saved_state_path):
                print('load successful')
                state = torch.load(saved_state_path, map_location=torch.device(device))
                test_model.load_state_dict(state['model_state_dict'])
            test_model.to(device)
        else:
            test_model = model
        test_model.eval()

        test_dataloader = DataLoader(validate_dataset, batch_size=5)
        train_dataloader = DataLoader(train_dataset, batch_size=10)
        false_positive_roc= [[] for _ in range(11)]
        true_positive_roc = [[] for _ in range(11)]

        with torch.no_grad():
            train_correct = 0
            correct = 0
            false_positive = 0
            false_negative = 0
            total = 0
            train_total = 0
            total_negative = 0
            total_positive = 0
            test_true_positive = 0

            if eval:
                for index, data in enumerate(train_dataloader):

                    images = data[0].to(device)
                    label = data[1].to(device)
                    output = test_model(images)
                    output = nn.Sigmoid()(output)

                    predict = (output >= 0.5).long()
                    predict = torch.squeeze(predict)
                    train_total += len(label)

                    train_correct += (predict == label).sum().item()
            else:
                train_total += 1

            for index, data in enumerate(test_dataloader):
                images = data[0].to(device)
                label = data[1].to(device)
                output = test_model(images)
                output = nn.Sigmoid()(output)

                predict = (output >= 0.5).long()
                predict = torch.squeeze(predict)
                total += len(label)

                correct += (predict == label).sum().item()
                wrong_prediction = predict != label
                positive_label = label == 1
                negative_label = label == 0
                false_positive += torch.logical_and(wrong_prediction, negative_label).sum().item()
                false_negative += torch.logical_and(wrong_prediction, positive_label).sum().item()
                temp_total_positive = positive_label.sum().item()
                temp_total_negative = negative_label.sum().item()
                total_positive += temp_total_positive
                total_negative += temp_total_negative
                test_true_positive += positive_label.sum().item()

                for p in range(11):

                    roc_predict = (output >= 0.1 * p).long()
                    roc_predict = torch.squeeze(roc_predict)
                    roc_wrong = roc_predict != label
                    roc_false_positive = torch.logical_and(roc_wrong, negative_label).sum().item()
                    roc_false_negative = torch.logical_and(roc_wrong, positive_label).sum().item()

                    roc_false_positive_rate = roc_false_positive / max(temp_total_negative,1)
                    roc_true_positive_rate = 1 - (roc_false_negative / max(temp_total_positive,1))
                    if temp_total_negative != 0:
                        false_positive_roc[p].append(roc_false_positive_rate)
                    if temp_total_positive != 0:
                        true_positive_roc[p].append(roc_true_positive_rate)


            if not cross_validate:
                print('Accuracy of the model is {}%'.format(100 * correct / total))
        print('Baseline accuracy is', test_true_positive / total)
        true_positive_roc = [sum(i)/len(i) for i in true_positive_roc]
        false_positive_roc = [sum(i) / len(i) for i in false_positive_roc]
        return train_correct / train_total, correct / total, false_positive / total_negative, \
               false_negative / total_positive, true_positive_roc, false_positive_roc
    return validate