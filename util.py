
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
            args.m.upload(join(args.output_folder, curr['a']['n']), args.mega_folder[0])
            args.m.destroy(curr['h'])
            
    #if folder was empty, populate it
    if i == 0:
        args.m.upload(join(args.output_folder, "debug.log"), args.mega_folder[0])
        args.m.upload(join(args.output_folder, "info.log"), args.mega_folder[0])
        args.m.upload(join(args.output_folder, "last_model.pth"), args.mega_folder[0])
        if is_best:
            args.m.upload(join(args.output_folder, "best_model.pth"), args.mega_folder[0])

def upload_from_local(src_folder, dest_folder, is_best, mega_username = "c.blangio@gmail.com"):
    mega = Mega()
    pwd = getpass.getpass(f"insert password for mega account ({mega_username}):")
    m = mega.login(mega_username, pwd)
    mega_folder = MyFind(m, dest_folder)
    files = m.get_files_in_node(mega_folder[0]) #this is a dictionary
    i=0
    for key in files:
        i+=1
        curr = files[key]
        #don't overwrite best model
        if curr['a']['n'] != "best_model.pth" or is_best:
            m.upload(join(src_folder, curr['a']['n']), mega_folder[0])
            m.destroy(curr['h'])
            
    #if folder was empty, populate it
    if i == 0:
        m.upload(join(src_folder, "debug.log"), mega_folder[0])
        m.upload(join(src_folder, "info.log"), mega_folder[0])
        m.upload(join(src_folder, "last_model.pth"), mega_folder[0])
        if is_best:
            m.upload(join(src_folder, "best_model.pth"), mega_folder[0])

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

            if isinstance(curr['a'], dict) == False:
                #this file does not have a name, we can't use it so continue
                continue
            
            #we check if this one is the right folder for our depth
            if curr['a']['n'] == folder and curr['p'] == parentid:
                parentid = curr['h']
                target = curr
                break
    
    if isinstance(curr['a'], dict) == False:
        #this file doesn't even have a name, definitely not the one we're looking for
        return None

    if curr['a']['n'] != folders[len(folders)-1]:
      return None
    return (target['h'], target)
    
