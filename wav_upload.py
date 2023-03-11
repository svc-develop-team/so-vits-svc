from google.colab import files
import shutil
import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, help="type of file to upload")
    args = parser.parse_args()
    file_type = args.type

    basepath = os.getcwd()
    uploaded = files.upload() # 上传文件
    assert(file_type in ['zip', 'audio'])
    if file_type == "zip":
        upload_path = "./upload/"
        for filename in uploaded.keys():
            #将上传的文件移动到指定的位置上
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, "userzip.zip"))
    elif file_type == "audio":
        upload_path = "./raw/"
        for filename in uploaded.keys():
            #将上传的文件移动到指定的位置上
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, filename))