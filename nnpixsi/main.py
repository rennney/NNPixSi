#!/usr/bin/env python3
'''
A high level interface .
'''
from pathlib import Path

class Main:
    '''
    Main entry point (almost).

    See also __main__
    '''

    def __init__(self, instore, outstore=None):
        '''
        Fill with input/output file interface
        '''
        self.instore_path = Path(instore).resolve()




        
