import os
import re
import logging
from time import sleep
from threading import (
    Thread,
    Event
)
from guizero import (
    App,
    Box,
    Text,
    TextBox,
    PushButton,
    select_folder
)
from DL_model.predict_imgs import DLModel
from root_tips.scenario2 import root_tip_analysis


def pipeline_run(input_path, output_folder, results_path, root_tip_size, exit_flag=None):
    """
    Make image segmentation and compute root tip growth, save results.
    """
    if not os.path.exists(input_path):
        raise ValueError(f"Input path is invalid! Directory {input_path} does not exist")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    dl_images = f'{output_folder}/DL_images'
    if not os.path.exists(dl_images):
        os.mkdir(dl_images)
    if os.listdir(dl_images):
        logging.warning('Folder for processed images is not empty and any duplicate images will be overwritten. Use CTRL-C to abort.')
        sleep(5)

    DLModel().apply_dl_model(input_path, dl_images, exit_flag)
    if exit_flag and exit_flag.is_set():
        return

    root_tip_images = f'{output_folder}/tip_images'
    if not os.path.exists(root_tip_images):
        os.mkdir(root_tip_images)
    root_tip_results = f'{output_folder}/tip_daily_changes'
    if not os.path.exists(root_tip_results):
        os.mkdir(root_tip_results)

    prev_pred_path = None
    for img_name in os.listdir(input_path):
        if exit_flag and exit_flag.is_set():
            logging.info("Received exit signal, terminating run")
            return
        img_path = f'{input_path}/{img_name}'

        img_pred_name = f'{img_name.strip(".jpg")}-prediction.jpg'
        img_pred_path = f'{dl_images}/{img_pred_name}'

        if prev_pred_path:
            day = re.search('20[0-9][0-9]\.[0-1][0-9]\.[0-3][0-9]', img_name).group(0)
            df = root_tip_analysis(
                prev_pred_path,
                img_pred_path,
                img_path,
                root_tip_size,
                root_tip_images,
                root_tip_results,
                day
            )
            #TODO save average root growth to save in aggregate results

        prev_pred_path = img_pred_path


    with open(results_path, 'w') as out:
        logging.info(f"Writing results to {results_path}")
        out.write('RESULTS\n') # TODO

class RootGrowthGUI():
    thread = None
    output_user_set = False
    exit_flag = Event()

    def __init__(self, defaults):
        self.app = App('Root image analyzer', width=700, height=300, layout='grid')

        self.cwd = os.getcwd()
        self.input_path = defaults['INPUT']
        self.output_dir_path = f'{defaults["INPUT"].strip("/").split("/")[-1]}--processed/'

        Text(self.app, text='Select image folder', grid=[0,0], align='left')
        input_button = PushButton(self.app, command=self.choose_input, text='Choose', grid=[1,0], align='left')
        self.input_path_text = Text(self.app, text=self.input_path, grid=[2,0])

        Text(self.app, text='Set minimum root tip size searched (mm)', grid=[0,1], align='left')
        self.tip_size = TextBox(self.app, text=defaults['TIP_SIZE'], grid=[1,1], align='left')

        Text(self.app, text='Name output file', grid=[0,2], align='left')
        self.output_file_box = TextBox(self.app, text=defaults['OUTPUT'].strip('./'), grid=[1,2], align='left')

        Text(self.app, text='Choose directory for processed images and data', grid=[0,3], align='left')
        output_dir_button = PushButton(self.app, command=self.choose_out_dir, text='Choose', grid=[1,3], align='left')
        self.output_dir_text = Text(self.app, text=self.output_dir_path, grid=[2,3])

        start_button = PushButton(self.app, command=self.start_model, text='Start', grid=[1,5], align='left')
        exit_button = PushButton(self.app, command=self.exit, text='Exit', grid=[2,5], align='left')

        self.app.display()


    def choose_input(self):
        self.input_path = select_folder(title='Select folder', folder=self.cwd)
        self.input_path_text = Text(self.app, text=self.input_path, grid=[2,0])
        if not self.output_user_set:
            self.output_dir_path = f'{self.input_path.split("/")[-1]}--processed'
            self.output_dir_text = Text(self.app, text=self.output_dir_path, grid=[2,3])


    def choose_out_dir(self):
        self.output_dir_path = select_folder(title='Select folder', folder=self.cwd)
        self.output_dir_text = Text(self.app, text=self.output_dir_path, grid=[2,2])
        self.output_user_set = True


    def start_model(self):
        if self.thread and self.thread.is_alive():
            return
        output = self.output_file_box.value
        size = self.tip_size.value
        self.thread = Thread(
            target=pipeline_run,
            args=(self.input_path, self.output_dir_path, output, size, self.exit_flag)
        )
        self.thread.start()


    def exit(self):
        self.exit_flag.set()
        self.app.destroy()
