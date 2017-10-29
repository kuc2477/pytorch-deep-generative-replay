#!/usr/bin/env python3
from contextlib import contextmanager
import os
import os.path
import argparse
import subprocess
import zipfile
from colorama import Fore
from tqdm import tqdm
import requests


# ================
# Helper Functions
# ================

def c(string, color):
    return '{}{}{}'.format(getattr(Fore, color.upper()), string, Fore.RESET)


@contextmanager
def log(start, end, start_color='yellow', end_color='cyan'):
    print(c('>> ' + start, start_color))
    yield
    print(c('>> ' + end, end_color) + '\n')


def _download(url, filename=None):
    local_filename = filename or url.split('/')[-1]
    temp_filename = '.{}'.format(local_filename)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(temp_filename, 'wb') as f:
        for chunk in tqdm(
                response.iter_content(1024 * 32),
                total=total_size // (1024 * 32),
                unit='KiB', unit_scale=True,
        ):
            if chunk:
                f.write(chunk)
    response.close()
    os.rename(temp_filename, local_filename)
    return local_filename


def _extract_zip(zipfile_path, extraction_path='.'):
    with zipfile.ZipFile(zipfile_path) as zf:
        extracted_dirname = zf.namelist()[0]
        zf.extractall(extraction_path)
    return extracted_dirname


def _extract_gz(gzfile_path, extraction_path='.'):
    cmd = ['gzip', '-d', gzfile_path]
    subprocess.call(cmd)
    return '.'.join(gzfile_path.split('.')[:-1])


def _download_zip_dataset(
        url, dataset_dirpath, dataset_dirname, download_path=None):
    download_path = _download(url)
    download_dirpath = os.path.dirname(download_path)
    extracted_dirname = _extract_zip(download_path)

    os.remove(download_path)
    os.renames(os.path.join(download_dirpath, extracted_dirname),
               os.path.join(dataset_dirpath, dataset_dirname))


def _download_gz_dataset(
        url, dataset_dirpath, dataset_dirname, download_path=None):
    download_path = _download(url)
    download_dirpath = os.path.dirname(download_path)
    extracted_filename = _extract_gz(download_path)

    os.renames(os.path.join(download_dirpath, extracted_filename),
               os.path.join(dataset_dirpath, dataset_dirname,
                            extracted_filename))


# ====
# Main
# ====

def maybe_download_lsun(dataset_dirpath, dataset_dirname,
                        category, set_name, tag='latest'):
    dataset_path = os.path.join(dataset_dirpath, dataset_dirname)
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
        '&category={category}&set={set_name}'.format(**locals())

    # check existance
    if os.path.exists(dataset_path):
        print(c(
            'lsun dataset already exists: {}'
            .format(dataset_path), 'red'
        ))
        return

    # start downloading lsun dataset
    with log(
            'download lsun dataset from {}'.format(url),
            'downloaded lsun dataset to {}'.format(dataset_path)):
        _download_zip_dataset(url, dataset_dirpath, dataset_dirname)


parser = argparse.ArgumentParser(description='LSUN dataset downloading CLI')
parser.add_argument('--dataset-dir', type=str, default='./datasets/lsun')
parser.add_argument('--category', type=str, default='bedroom')
parser.add_argument('--test', action='store_false', dest='train')


if __name__ == '__main__':
    args = parser.parse_args()
    category = args.category
    set_name = 'train' if args.train else 'test'
    maybe_download_lsun(
        args.dataset_dir,
        '{category}_{set_name}'.format(category=category, set_name=set_name),
        category,
        set_name,
    )
