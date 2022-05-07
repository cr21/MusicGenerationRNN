from Config import Config
import os
music_notes = ""


def read_files(dir=Config.DATA_DIR):
    music_notes =""
    for root, dirs, files in os.walk(dir, topdown=False):
        for name in files:
            full_path = os.path.join(root, name)
            print("Reading file", full_path)
            with open(full_path, 'r',encoding='windows-1252') as f:
                music_notes +=f.read()
    print("Total Number of characters in all the music notes combined are : {}".format(len(music_notes)))
    return music_notes


