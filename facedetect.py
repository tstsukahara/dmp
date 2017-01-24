import cv2
import sys
import os

def main():
    # イメージファイル一覧を取得する
    filedir= "/Users/tsuyoshi/Desktop/dmp"
    files = os.listdir(filedir)

    # 顔認識させる
    for file in files:
        # 拡張子を確認し画像ファイルの場合のみ実行
        path, ext = os.path.splitext(file)
        if ext == ".png" or ext == ".jpg":
            # detect_face("%s/%s" % (filedir, file))
            trim_face("%s/%s" % (filedir, file))

def detect_face(image_file):
    # カスケードファイルのパスを指定 --- (※1)
    cascade_file = "haarcascade_frontalface_alt.xml"

    # 画像の読み込み --- (※2)
    image = cv2.imread(image_file)
    # グレースケールに変換
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔認識用特徴量ファイルを読み込む --- (※3)
    cascade = cv2.CascadeClassifier(cascade_file)
    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(150,150))

    if len(face_list) > 0:
        # 認識した部分を囲む --- (※4)
        print(face_list)
        color = (0, 0, 255)
        for face in face_list:
            x,y,w,h = face
            cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness=8)
        # 描画結果をファイルに書き込む --- (※5)
        cv2.imwrite("%s_faces.png" % image_file, image)
    else:
        print("no face")

def trim_face(image_file):
    # ファイルパス取得
    path = os.path.dirname(image_file)
    name, _ = os.path.splitext( os.path.basename(image_file))
    print("path=%s" % path)
    print("name=%s" % name)

    # カスケードファイルのパスを指定 --- (※1)
    cascade_file = "haarcascade_frontalface_alt.xml"

    # 画像の読み込み --- (※2)
    image = cv2.imread(image_file)
    # グレースケールに変換
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔認識用特徴量ファイルを読み込む --- (※3)
    cascade = cv2.CascadeClassifier(cascade_file)
    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(150,150))

    #ディレクトリの作成
    if len(face_list) > 0:
        path_face = path + '/face'
        if os.path.isdir(path_face):
            pass
        else:
            os.mkdir(path_face)

    i = 0;
    for face in face_list:
        #顔だけ切り出して保存
        x = face[0]
        y = face[1]
        width = face[2]
        height = face[3]
        trmimage = image[y:y+height, x:x+width]
        new_image_path = path_face + '/' + name + '_' + str(i) + '.png'
        print("new_image_path=%s" % new_image_path)
        cv2.imwrite(new_image_path, trmimage)
        i += 1

if __name__ == '__main__':
    main()
