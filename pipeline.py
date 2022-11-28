from train import *
from util import get_useable_image_and_label, generate_mask_area_given_img, calculate_var
from validate import *
import torch.nn as nn
from dataset import *
from datetime import datetime



def multi_CrossFoldPipeline(job_configs):
    cur_time = datetime.now()
    str_cur_time = str(cur_time).replace(".","-").replace(":","-")
    print("Number of run group is {}".format(len(job_configs)))

    for run in range(len(job_configs)):
        for group_run in range(len(job_configs[run])):
            false_positive_rate, true_positive_rate = CrossFoldPipeline(job_configs[run][group_run])
            if job_configs[run][group_run]["roc"]:

                np.save("./roc_raw/{}-fp-{}.npy".format(job_configs[run][group_run]["model_name"], str_cur_time), false_positive_rate)
                np.save("./roc_raw/{}-tp-{}.npy".format(job_configs[run][group_run]["model_name"], str_cur_time), true_positive_rate)
                plt.plot(false_positive_rate, true_positive_rate, label=job_configs[run][group_run]["model_name"])

        x = [0, 1]
        y = [0, 1]
        plt.plot(x, y, label="random prediction", linestyle="dashed")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.legend()
        plt.savefig("./out_plot/run_group-{}-{}.png".format(job_configs[run][0]["run_group"], str_cur_time), dpi=250)
        plt.close()

def CrossFoldPipeline(job_config):

    flipping = job_config["flipping"]
    learning_rate = job_config["learning_rate"]
    masking = job_config["masking"]
    num_run = job_config["num_run"]
    channel = job_config["channel"]
    roc = job_config["roc"]
    draw = job_config["draw"]
    epochs = job_config["epochs"]
    model_name = job_config["model_name"]
    model = job_config["model"]

    image_names, labels = get_useable_image_and_label(filepath='./Master IOL 700 Study Data.xlsx')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5]).to(device))

    if channel == 1:
        image_stage1_2_transforms = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.ToTensor(),
            T.Resize((158, 320)),
            *generate_mask_area_given_img((158, 320), 3, masking)
        ])
    else:
        image_stage1_2_transforms = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Resize((158, 320)),
            *generate_mask_area_given_img((158, 320), 3, masking)
        ])

    train_accuracy_list = []
    validation_accuracy_list = []
    false_positive_list = []
    false_negative_list = []
    AUC_result_list = []
    false_positive_roc_list = []
    true_positive_roc_list = []

    result_saving_folder = "./result/"
    if flipping:
        print("USE FLIPPED IMAGE")
        image_path = "./cropped_img_aligned/"
        save_path = result_saving_folder + model_name + "_" + str(learning_rate) + "-Flipped" +".txt"
    else:
        print("USE ORIGINAL IMAGE")
        image_path = "./cropped_image/"
        save_path = result_saving_folder + model_name + "_" + str(learning_rate) + "-Original" + ".txt"
    # Stage 1 Code Example

    final_save_path = result_saving_folder + job_config["save_path"] if "save_path" in job_config else save_path

    print("Save Path is {}".format(final_save_path))


    for i in range(num_run):
        print(i, 'th training test')
        Stage1Dataset = IOLClassfierDataset(image_names, labels, image_path=image_path,
                                            transform=image_stage1_2_transforms)
        stage1_train = train_stage1(epochs, learning_rate, 13, loss, cross_validate=True, device=device)
        stage1_validate = validate_stage1(cross_validate=True, device=device)
        train_accuracy, va, fp, fn, auc, fp_ROC, tp_ROC = crossfold_validation(model, stage1_train, stage1_validate,
                                                               Stage1Dataset, ROC=roc, draw=draw, device=device)

        print(train_accuracy, va, fp, fn, auc)
        train_accuracy_list.append(train_accuracy)
        validation_accuracy_list.append(va)
        false_positive_list.append(fp)
        false_negative_list.append(fn)
        AUC_result_list.append(auc)
        false_positive_roc_list.append(fp_ROC)
        true_positive_roc_list.append(tp_ROC)


    print('Average accuracy')

    if roc:
        print(sum(train_accuracy_list) / len(train_accuracy_list),
              sum(validation_accuracy_list) / len(validation_accuracy_list),
              sum(false_positive_list) / len(false_positive_list),
              sum(false_negative_list) / len(false_negative_list),
              sum(AUC_result_list) / len(AUC_result_list))
        print(calculate_var(train_accuracy_list),
              calculate_var(validation_accuracy_list),
              calculate_var(false_positive_list),
              calculate_var(false_negative_list),
              calculate_var(AUC_result_list))

    else:
        print(sum(train_accuracy_list) / len(train_accuracy_list),
              sum(validation_accuracy_list) / len(validation_accuracy_list),
              sum(false_positive_list) / len(false_positive_list),
              sum(false_negative_list) / len(false_negative_list))
        print(calculate_var(train_accuracy_list),
              calculate_var(validation_accuracy_list),
              calculate_var(false_positive_list),
              calculate_var(false_negative_list))



    with open(final_save_path , "w") as file:
        file.write("train_accuracy is {}+- {} \n".format(sum(train_accuracy_list)/len(train_accuracy_list), calculate_var(train_accuracy_list)))
        file.write("validation_accuracy is {}+- {} \n".format(sum(validation_accuracy_list)/len(validation_accuracy_list), calculate_var(validation_accuracy_list)))
        file.write("false_positive_rate is {}+- {} \n".format(sum(false_positive_list)/len(false_positive_list), calculate_var(false_positive_list)))
        file.write("false negative rate is {}+- {} \n".format(sum(false_negative_list)/len(false_negative_list), calculate_var(false_negative_list)))
        if roc:
            file.write("AUC is {}+- {} \n".format(sum(AUC_result_list) / len(AUC_result_list), calculate_var(AUC_result_list)))

    return np.average(false_positive_roc_list, axis=0), np.average(true_positive_roc_list, axis=0)