import os
import os.path
import hashlib
import errno
import requests
import urllib.request
from tqdm import tqdm

# 生成 tqdm 进度条更新器
def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

# 检查文件完整性（根据 md5）
def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # 以 1MB 分块读取
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    return md5c == md5

# 创建目录（兼容 Python2）
def makedir_exist_ok(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# 从 URL 下载文件
def download_url(url, root, filename=None, md5=None):
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    makedir_exist_ok(root)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('已存在且校验通过: ' + fpath)
    else:
        try:
            print('下载中 ' + url + ' 到 ' + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
        except OSError:
            if url.startswith('https'):
                url = url.replace('https:', 'http:')
                print('HTTPS 失败，尝试 HTTP: ' + url)
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())

# 列出目录下的所有子目录
def list_dir(root, prefix=False):
    root = os.path.expanduser(root)
    directories = [
        os.path.join(root, d) if prefix else d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    return directories

# 列出目录下所有指定后缀的文件
def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = [
        os.path.join(root, f) if prefix else f
        for f in os.listdir(root)
        if os.path.isfile(os.path.join(root, f)) and f.endswith(suffix)
    ]
    return files

# 从 Google Drive 下载文件
def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    url = "https://docs.google.com/uc?export=download"
    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('已存在且校验通过: ' + fpath)
    else:
        session = requests.Session()
        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, fpath)

# 提取下载确认令牌
def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

# 保存文件内容到本地
def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()
