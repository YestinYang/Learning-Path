import os

dir = "D:\prank"

def rename_file(dir):
    # 1) get the list of file name
    file_list = os.listdir(dir)  # r for do not modify the result

    # 1.5) handle working directory
    saved_dir = os.getcwd()
    os.chdir(dir)
    print ("Working directory is changed to " + dir)

    # 2) rename them
    for file_name in file_list:
        print ("Original name:" + file_name)
        print ("Renamed to:" + file_name.translate(None,"0123456789"))
        os.rename(file_name, file_name.translate(None,"0123456789"))

    # 2.5) back to original dir
    print ("Working directory is changed back to " + saved_dir)
    os.chdir(saved_dir)

rename_file(dir)