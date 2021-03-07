import logging
from os import path
from appJar import gui

LABELS = {
    'in_dir': 'Input directory',
    'out_file': 'Output file',
    'start_button': 'Run model',
    'exit_button': 'Exit'
}

def mock_model_run(input_path, output_path):
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
        self.defaults = defaults

        self.app = gui()
        self.app.addDirectoryEntry(LABELS['in_dir'])

        self.app.addLabelEntry(LABELS['out_file'])
        default_out = defaults['DEFAULT_OUTPUT'].strip('./')
        self.app.setEntryDefault(LABELS['out_file'], f'{default_out}')

        self.app.addButton(LABELS['start_button'], self.start_model)
        self.app.addButton(LABELS['exit_button'], self.exit)

    def start_model(self):
        input_path = self.app.entry(LABELS['in_dir'])
        if not input_path:
            input_path = self.defaults['DEFAULT_INPUT']

        output_path = self.app.entry(LABELS['out_file'])
        if not output_path:
            output_path = self.defaults['DEFAULT_OUTPUT']

        mock_model_run(input_path, output_path)

    def exit(self):
        self.app.stop()

    def run(self):
        self.app.go()