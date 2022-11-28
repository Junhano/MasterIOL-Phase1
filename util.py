import os
import math
import pandas as pd
from torch.utils.data import Subset, random_split
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torchvision.models as models
from model import LaplacianNN, EdgeDetectionNN, BaselineNN, BaselineCNN, ClassifierPlusPretrain

from transform import SpecificErase

DEFAULT_FILEPATH = './Master IOL 700 Study Data.xlsx'


def generate_pretrain_model():
    model_ft = models.vgg16(pretrained=True)
    feature_weight = model_ft.features[:9]
    for param in feature_weight.parameters():
        param.require_grad = False

    return ClassifierPlusPretrain(feature_weight, 128)

def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_useable_image_and_label(sheet_name = 0, filepath = DEFAULT_FILEPATH):
    '''
    Function to get all the useable image and label based on the Study data and it's
    availability (When it exist in image database)
    :param filepath:
    :return:
    '''



    research_data = pd.read_excel(filepath, sheet_name=sheet_name, engine='openpyxl')


    useable_row = research_data[research_data['Useable'] == 1.0]
    image_names = useable_row['Image name '].values.tolist()
    success = useable_row['OVERALL success or failure'].values.tolist()


    labels = [1 if i.upper() == 'SUCCESS' else 0 for i in success]


    process_image_list = []
    process_label_list = []


    for _, _, file_names in os.walk('./cropped_image'):
        for i in range(len(image_names)):
            if image_names[i] + '.jpg' in file_names:
                process_image_list.append(image_names[i] + '.jpg')
                process_label_list.append(labels[i])
    positive_label = 0
    for i in process_label_list:
        if i == 1:
            positive_label += 1
    print('Positive Label Percentage is', positive_label / len(process_label_list))
    print(len(process_label_list))


    return process_image_list, process_label_list

def get_fake_useable_image_and_label(positive_percentage, sheet_name=0, filepath=DEFAULT_FILEPATH):
    '''

    :param filepath:
    :return:
    '''

    research_data = pd.read_excel(filepath, sheet_name=sheet_name, engine='openpyxl')


    useable_row = research_data[research_data['Useable'] == 1.0]
    image_names = useable_row['Image name '].values.tolist()



    process_image_list = []

    for _, _, file_names in os.walk('./cropped_image'):
        for i in range(len(image_names)):
            if image_names[i] + '.jpg' in file_names:
                process_image_list.append(image_names[i] + '.jpg')

    random_list = np.random.uniform(0, 1, size = len(process_image_list))

    random_filter = random_list < positive_percentage
    process_label_list = list(random_filter)

    positive_label = 0
    for i in process_label_list:
        if i == 1:
            positive_label += 1
    print('Positive Label Percentage is', positive_label / len(process_label_list))
    print(len(process_image_list))

    return process_image_list, process_label_list


def random_split_ds(dataset, kfold, stradified, class_index=1):

    if stradified:
        index_representation = []
        class_representation = []

        for index in range(len(dataset)):
            index_representation.append(index)
            class_representation.append(dataset[index][class_index])

        skf = StratifiedKFold(n_splits=kfold, shuffle=True)
        skf_split = skf.split(index_representation, class_representation)

        split_index = []
        for i in skf_split:
            split_index.append(i[1])

        return [Subset(dataset, indexes) for indexes in split_index]


    else:
        total_length = len(dataset)
        full_split_len = total_length // (kfold)
        extra_len = full_split_len + total_length % kfold

        split_len_list = [full_split_len for _ in range(kfold - 1)]
        split_len_list.append(extra_len)

        return random_split(dataset, split_len_list)


def temporal_nasal_align(source_path = './cropped_image', dest_path = "./cropped_img_aligned"):
    source_dir = Path(source_path)
    dest_dir = Path(dest_path)
    dest_dir.mkdir(exist_ok = True)
    if source_dir.exists() and dest_dir.exists():
        for p in source_dir.iterdir():
            if p.is_file() and p.suffix == '.jpg':
                I = plt.imread(p)
                if "OD" in p.name:
                    I = np.flip(I, axis=1)
                plt.imsave(dest_dir / p.name, I)

    else:
        print("sourse or destination doesn't exist")



def generate_mask_area_given_img(imgshape, divide_times, masked_area):

    if len(masked_area) != 0 and max(masked_area) >= divide_times * divide_times:
        raise ValueError

    return_list = []
    height = math.ceil(imgshape[0] / divide_times)
    width = math.ceil(imgshape[1] / divide_times)
    for area in masked_area:
        column = area // divide_times
        row = area % divide_times
        if column == divide_times - 1:
            i = imgshape[0] - height
        else:
            i = column * height
        if row == divide_times - 1:
            j = imgshape[1] - width
        else:
            j = row * width
        return_list.append(SpecificErase(i, j, height, width))



    return return_list


def draw_roc(fp_ROCs, tp_ROCs, labels):

    for i in range(len(fp_ROCs)):
        plt.plot(np.average(fp_ROCs[i], axis=0), np.average(tp_ROCs[i], axis=0), label=labels[i])

    x = [0, 1]
    y = [0, 1]
    plt.plot(x, y, label="random prediction", linestyle="dashed")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    plt.show()

def read_config(config):

    config_dict = dict()

    config_dict["run_group"] = config["Run_group"]
    config_dict["model_name"] = config["Model_name"]
    config_dict["learning_rate"] = float(config["LearningRate"]) if "LearningRate" in config else 3e-4
    config_dict["flipping"] = config.getboolean("Flipping") if "Flipping" in config else True
    config_dict["num_run"] = int(config["Num_run"]) if "Num_run" in config else 15
    config_dict["masking"] = config["Masking"].split(",") if "Masking" in config else []
    config_dict["roc"] = config.getboolean("ROC") if "ROC" in config else False
    config_dict["draw"] = config.getboolean("Draw") if "Draw" in config else False
    config_dict["epochs"] = int(config["Epochs"]) if "Epochs" in config else 10
    config_dict["disable"] = config.getboolean("Disable") if "Disable" in config else False

    if "Save_path" in config:
        config_dict["save_path"] = config["Save_path"]

    exist_model = {
        "LaplacianNN": LaplacianNN,
        "EdgeDetectionNN": EdgeDetectionNN,
        "BaselineNN": BaselineNN,
        "BaselineCNN": BaselineCNN,
    }
    model = config["model"]
    if model == "PretrainNN":
        config_dict["model"] = generate_pretrain_model()
        config_dict["channel"] = 3
    elif model in exist_model.keys():
        config_dict["model"] = exist_model[model]
        config_dict["channel"] = 1
    else:
        raise KeyError("Model which specify in config.ini is not defined")
    return config_dict


def calculate_var(varlist):
    avg = sum(varlist) / len(varlist)
    return sum((x - avg) ** 2 for x in varlist) / len(varlist)


def generate_train_configs(configpath = 'config.ini'):
    import configparser

    config = configparser.ConfigParser()
    config.read(configpath)

    train_config_run_group = []
    train_config_run_group_dict = dict()

    for section in config.sections():
        config_dict = read_config(config[section])
        if not config_dict["disable"]:
            run_group = config_dict["run_group"]
            if run_group in train_config_run_group_dict:
                train_config_run_group[train_config_run_group_dict[run_group]].append(config_dict)
            else:
                train_config_run_group.append([])
                train_config_run_group[-1].append(config_dict)
                train_config_run_group_dict[run_group] = len(train_config_run_group) - 1
    return train_config_run_group