# -*- coding: utf-8 -*-

import sys
import os
import predict
import facedetect

def main():
    argvs = sys.argv
    if (len(argvs) != 2):
        print('Usage: # python %s <file_path>' % argvs[0])
        quit()

    # 画像から顔を切り出した画像ファイルを作成する
    file_path = argvs[1]
    path_face_trim, trim_faces = facedetect.trim_face(file_path)

    if len(trim_faces) == 0:
        print('No face has detected.')
        quit()

    # 分類モデルに顔画像を投入する
    answer = predict.test(path_face_trim)

    # 結果を元画像に埋め込んで表示する
    detect_file = facedetect.display_face(file_path, answer, trim_faces)
    os.system('open %s' % detect_file)


if __name__ == "__main__":
    main()
