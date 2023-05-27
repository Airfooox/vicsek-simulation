import os
import shutil


def copyAndRenameFolders(sourceDir, constant):
    destinationDir = os.path.join(sourceDir, '_renamed')
    os.makedirs(destinationDir, exist_ok=True)

    for root, dirs, files in os.walk(sourceDir):
        if '_renamed' in dirs:
            dirs.remove('_renamed')

        for dirName in dirs:
            folderName = dirName.split('_')
            if len(folderName) != 2:
                continue

            int1, int2 = folderName
            try:
                int1 = int(int1)
                int2 = int(int2)
            except ValueError:
                continue

            newInt2 = int2 + constant
            newFolderName = f"{int1}_{newInt2}"
            sourcePath = os.path.join(root, dirName)
            destinationPath = os.path.join(destinationDir, newFolderName)
            shutil.copytree(sourcePath, destinationPath)
            print(f"Folder '{dirName}' copied and renamed to '{newFolderName}'.")


# Example usage
sourceDirectory = r"E:\simulationdata\multiple_groups\nematic\va_over_eta_multiple_groups_58fc0010-5052-46e8-b0d0-b2ad976a489b"
constant = 100

copyAndRenameFolders(sourceDirectory, constant)
