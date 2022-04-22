# Program's Function: Appends the folder id to the names of  all the jpgs in CoastSeg's data directory
# Author: Sharon Fitzpatrick
# Date: 4/21/2022
import os
import shutil
import glob


def rename_jpgs(src_path: str) -> None:
    """ Renames all the jpgs in the data directory in CoastSeg
    Args:
        src_path (str): full path to the data directory in CoastSeg
    """
    files_renamed = False
    for folder in os.listdir(src_path):
        folder_path = src_path + os.sep + folder
        # Split the folder name at the first _
        folder_id = folder.split("_")[0]
        folder_path = folder_path + os.sep + "jpg_files" + os.sep + "preprocessed"
        jpgs = glob.glob1(folder_path + os.sep, "*jpg")
        # Append folder id to basename of jpg if not already there
        for jpg in jpgs:
            if folder_id not in jpg:
                files_renamed = True
                base, ext = os.path.splitext(jpg)
                new_name = folder_path + os.sep + base + "_" + folder_id + ext
                old_name = folder_path + os.sep + jpg
                os.rename(old_name, new_name)
        if files_renamed:
            print(f"Renamed files in {src_path} ")


def copy_files_to_dst(src_path: str, dst_path: str, glob_str: str) -> None:
    """copy_files_to_dst copies all the files from src_path to dest_path
    Args:
        src_path (str): full path to the data directory in CoastSeg
        dst_path (str): full path to the images directory in Sniffer
    """
    if not os.path.exists(dst_path):
        print(f"dst_path: {dst_path} doesn't exist.")
    elif not os.path.exists(src_path):
        print(f"src_path: {src_path} doesn't exist.")
    else:
        for file in glob.glob(glob_str):
            shutil.copy(file, dst_path)
        print(f"Copied files that matched {glob_str}  to {dst_path}")


if __name__ == "__main__":
    # STEP 1: Renamed jpgs in CoastSeg then Copy to Sniffer
    # ------------------------------------------------------------
    src_path = r"C:\1_USGS\1_CoastSeg\1_official_CoastSeg_repo\CoastSeg\data"
    rename_jpgs(src_path)
    # # Copy jpgs from CoastSeg's data directory to Sniffer
    dst_path = r"C:\1_USGS\1_CoastSeg\2_Sniffer\Sniffer\images"
    glob_str = src_path + str(os.sep + "**" + os.sep) * 3 + "*jpg"
    copy_files_to_dst(src_path, dst_path, glob_str)
    # ---------------------------------------------------------------

    # STEP 2: Copy jpgs and csv from Sniffer to the Sniffer-Classifier
    # ( Ensure the ONLY csv in Sniffer is the one that corresponds to your imagery)
    # ------------------------------------------------------------
    # # Copy jpgs from sniffer to sniffer-classifier
    src_path = r"C:\1_USGS\1_CoastSeg\2_Sniffer\Sniffer\images"
    dst_path = r"C:\1_USGS\2_Machine_Learning\3_sniffer_classifier\images"
    glob_str = src_path + os.sep + "*jpg"
    copy_files_to_dst(src_path, dst_path, glob_str)

    # # Copy the csv file from sniffer to sniffer-classifier's csv folder
    src_path = r"C:\1_USGS\1_CoastSeg\2_Sniffer\Sniffer"
    dst_path = r"C:\1_USGS\2_Machine_Learning\3_sniffer_classifier\csv"
    glob_str = src_path + os.sep + "*csv"
    copy_files_to_dst(src_path, dst_path, glob_str)
    # ------------------------------------------------------------------
