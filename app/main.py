import argparse
import logging
from sys import argv
from os import environ

from root_growth_gui import (RootGrowthGUI, mock_model_run)

DEFAULTS = {
    'DEFAULT_INPUT': './images/',
    'DEFAULT_OUTPUT': './results.csv'
}

class AppController():
    def __init__(self):
        self.arg_parser = argparse.ArgumentParser()
        default_in = DEFAULTS['DEFAULT_INPUT']
        default_out = DEFAULTS['DEFAULT_OUTPUT']
        self.arg_parser.add_argument(
            '--cli',
            dest='use_cli',
            action='store_true',
            help=f'Use --cli flag to run the program from the command line.'
        )
        self.arg_parser.add_argument(
            '--input',
            dest='input',
            default=default_in,
            type=str,
            help=f'Set path for input image directory. Default: {default_in}'
        )
        self.arg_parser.add_argument(
            '--output',
            dest='output',
            default=default_out,
            type=str,
            help=f'Set path for results output. Default: {default_out}'
        )

    def run(self):
        args = self.arg_parser.parse_args()
        if args.use_cli:
            mock_model_run(args.input, args.output)
        else:
            app = RootGrowthGUI(DEFAULTS)
            app.run()


if __name__ == '__main__':
    logging.basicConfig(level=environ.get('LOGLEVEL', 'INFO'))
    ctrl = AppController()
    ctrl.run()
