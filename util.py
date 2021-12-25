
import torch
import shutil
from os.path import join
from mega import Mega
import getpass


def save_checkpoint(args, state, is_best: bool, filename: str) -> None:
    model_path = join(args.output_folder, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.output_folder, "best_model.pth"))

def upload_checkpoint(args, is_best):
    mega = Mega()
    args.m = mega.login(args.mega_username, args.pwd)
    files = args.m.get_files_in_node(args.mega_folder[0]) #this is a dictionary
    i=0
    for key in files:
        i+=1
        curr = files[key]
        #don't overwrite best model
        if curr['a']['n'] != "best_model.pth" or is_best:
            args.m.destroy(curr['h'])
            args.m.upload(join(args.output_folder, curr['a']['n']), args.mega_folder[0])
    #if folder was empty, populate it
    if i == 0:
        args.m.upload(join(args.output_folder, "debug.log"), args.mega_folder[0])
        args.m.upload(join(args.output_folder, "info.log"), args.mega_folder[0])
        args.m.upload(join(args.output_folder, "last_model.pth"), args.mega_folder[0])
        if is_best:
            args.m.upload(join(args.output_folder, "best_model.pth"), args.mega_folder[0])

def init_mega(args):
    mega = Mega()
    args.pwd = getpass.getpass(f"insert password for mega account ({args.mega_username}):")
    args.m = mega.login(args.mega_username, args.pwd)

def init_tmp_dir(args):
    last = MyFind(args.m,join(args.load_from, "last_model.pth"))
    if last != None:
        args.m.download(last, args.output_folder)
    best = MyFind(args.m, join(args.load_from, "best_model.pth"))
    if best != None:
        args.m.download(best, args.output_folder)

#api function for retrieving a file from cloud given its full path was giving problems, so i made my own -_- 
def MyFind(m, filename):
    files = m.get_files()

    folders = filename.split("/")
    parentid = m.root_id
    target = {}

    #find the file (maybe not ideal as there doesn't seem to be a tree-like structure for the file system, but at least it works)
    for folder in folders:
        for file in files:
            curr = files[file] 
            #we check if this one is the right folder for our depth
            if curr['a']['n'] == folder and curr['p'] == parentid:
                parentid = curr['h']
                target = curr
                break

    return (target['h'], target)
    
