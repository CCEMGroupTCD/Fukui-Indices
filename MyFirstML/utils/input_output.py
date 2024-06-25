"""
Utilities for input and output
"""
import os

def write_to_yaml(dictionary, output_path, comment=None, verbose: bool=True):
    """
    Write a dictionary to yaml with a comment in the first line.
    """
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, "a") as file:
        if comment is not None:
            file.write('# "' + comment + '"\n')

        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")
        if verbose:
            print(f"Saved {len(dictionary)} entries to {os.path.basename(output_path)}.")

    return

def write_to_csv(df, output_path, comment, index=True, verbose: bool=True):
    """
    Write a df to csv with an comment in the first line.
    """
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, "a") as file:
        file.write('# "' + comment + '"\n')
        df.to_csv(file, index=index)
        if verbose:
            print(f"Saved {len(df)} entries to {os.path.basename(output_path)}.")

    return

def make_new_output_directory(rootdir, label) -> str:
    """
    Get numbered output directory `outdir` in parent directory `outdirname`.
    """
    os.makedirs(rootdir, exist_ok=True)

    # Get number that's not already used for a directory.
    num = 0
    dir_list = os.listdir(rootdir)
    while 'results_{}'.format(num) in '\t'.join(dir_list):
        num += 1

    outdir = os.path.join(rootdir, f'results_{num}_{label}')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    return outdir