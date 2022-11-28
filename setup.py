import os


def setup():
    '''
    RUN THIS FUNCTION TO SETUP REQUIRED FOLDER FOR RUNNING PIPELINE
    '''

    pwd = os.getcwd()

    result_out_folder = os.path.join(pwd, "result")
    out_plot_folder = os.path.join(pwd, "out_plot")
    roc_raw_data_folder = os.path.join(pwd, "roc_raw")


    if os.path.exists(result_out_folder):
        print("RESULT OUT FOLDER EXIST ALREADY")
    else:
        print("RESULT OUT FOLDER DOES NOT EXIST, CREATING RESULT OUT FOLDER")
        os.mkdir(result_out_folder)

    if os.path.exists(out_plot_folder):
        print("OUT PLOT FOLDER EXIST ALEADY")
    else:
        print("OUT PLOT FOLDER DOES NOT EXIST, CREATING OUT PLOT FOLDER FOLDER")
        os.mkdir(out_plot_folder)

    if os.path.exists(roc_raw_data_folder):
        print("ROC RAW DATA FOLDER EXIST ALREADY")
    else:
        print("ROC RAW DATA FOLDER DOES NOT EXIST, CREATING ROC RAW DATA FOLDER")
        os.mkdir(roc_raw_data_folder)




if __name__ == '__main__':
    setup()