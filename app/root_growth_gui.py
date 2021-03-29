from os import (
    getcwd,
    path
)
from guizero import (
    App,
    Box,
    Text,
    TextBox,
    PushButton,
    select_folder
)

def mock_model_run(input_path, output_path, output_images):
    """
    This is a placeholder function
    """
    if not path.exists(input_path):
        raise ValueError(f"Input path is invalid! Directory {input_path} does not exist")
    for i in range(5):
        logging.info(f'Running model... {(i+1)*20}%')
    with open(output_path, 'w') as out:
        logging.info(f"Writing results to {output_path}")
        out.write('RESULTS\n')

class RootGrowthGUI():

    def __init__(self, defaults):
        self.app = App('Root image analyzer', width=700, height=300, layout='grid')

        self.cwd = getcwd()
        self.input_path = defaults['INPUT']
        self.output_dir_path = defaults['OUTPUT_IMG']

        Text(self.app, text='Select image folder', grid=[0,0], align='left')
        input_button = PushButton(self.app, command=self.choose_input, text='Choose', grid=[1,0], align='left')
        self.input_path_text = Text(self.app, text=self.input_path, grid=[2,0])

        Text(self.app, text='Name output file', grid=[0,1], align='left')
        self.output_file_box = TextBox(self.app, text=defaults['OUTPUT'].strip('./'), grid=[1,1], align='left')
        Text(self.app, text='Choose directory for processed images', grid=[0,2], align='left')
        output_dir_button = PushButton(self.app, command=self.choose_out_dir, text='Choose', grid=[1,2], align='left')
        self.output_dir_text = Text(self.app, text=self.output_dir_path, grid=[2,2])

        start_button = PushButton(self.app, command=self.start_model, text='Start', grid=[1,4], align='left')
        exit_button = PushButton(self.app, command=self.exit, text='Exit', grid=[2,4], align='left')

        self.app.display()

    def choose_input(self):
        self.input_path = select_folder(title='Select folder', folder=self.cwd)
        self.input_path_text = Text(self.app, text=self.input_path, grid=[2,0])

    def choose_out_dir(self):
        self.output_dir_path = select_folder(title='Select folder', folder=self.cwd)
        self.output_dir_text = Text(self.app, text=self.output_dir_path, grid=[2,2])

    def start_model(self):
        output = self.output_file_box.value
        mock_model_run(self.input_path, output, self.output_dir_path)

    def exit(self):
        self.app.destroy()
