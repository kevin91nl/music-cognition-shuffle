import os
import random
import csv
import datetime

import shutil

blocks = [[file for file in os.listdir('data') if file.startswith('shuffle_elements_') and file.endswith('_%d.wav' % seed)] for seed in range(10)]

with open('experiment.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for seed in range(1, 3):
        random.seed(seed)
        files = blocks[seed]
        random.shuffle(files)
        writer.writerow([seed] + files)

        if not os.path.exists('experiment/%d' % seed):
            os.makedirs('experiment/%d' % seed)

        for index, file in enumerate(files):
            shutil.copy('data/%s' % file, 'experiment/%d/%d.wav' % (seed, index + 1))