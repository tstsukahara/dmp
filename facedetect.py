import cv2
import sys
import os
import shutil

NEMU = 0
RISA = 1
MIRIN = 2
EI = 3
PIN = 4
MOGA = 5

def main():
    # イメージファイル一覧を取得する
    filedir= "/Users/tsuyoshi/Desktop/dmp"
    files = os.listdir(filedir)

    # 顔認識させる
    for file in files:
        # 拡張子を確認し画像ファイルの場合のみ実行
        path, ext = os.path.splitext(file)
        if ext == ".png" or ext == ".jpg":
            detect_face("%s/%s" % (filedir, file))
            # trim_face("%s/%s" % (filedir, file))

def set_format(name):
    if name == NEMU:
        text = 'nemu'
        color = (213, 222, 180)
    elif name == RISA:
        text = 'risa'
        color = (222, 222, 222)
    elif name == MIRIN:
        text = 'mirin'
        color = (100, 100, 255)
    elif name == EI:
        text = 'ei'
        color = (50, 230, 230)
    elif name == PIN:
        text = 'pin'
        color = (220, 130, 50)
    elif name == MOGA:
        text = 'moga'
        color = (210, 125, 170)
    else:
        text = 'none'
        color = (0, 0, 0)

    return text, color

def display_face(image_file, label, face_list):
    # ファイルパス取得
    path = os.path.dirname(image_file)
    name, _ = os.path.splitext( os.path.basename(image_file))

    # 画像の読み込み
    image = cv2.imread(image_file)

    #ディレクトリの作成
    if len(face_list) > 0:
        path_face_rect = path + '/face_rect'
        if os.path.isdir(path_face_rect):
            shutil.rmtree(path_face_rect)
            os.mkdir(path_face_rect)
        else:
            os.mkdir(path_face_rect)

    rect_image_path = ''
    if len(face_list) > 0:
        # 認識した部分を囲む
        i = 0
        for face in face_list:
            x,y,w,h = face
            text, color = set_format(label[i])
            # 枠を作成する
            cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness=2)
            # cv2.putText(画像, 文字, 左下座標, フォント, 文字の大きさ, 色, 文字の太さ, 線の種類)
            cv2.putText(image, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 16)
            i += 1

        # 描画結果をファイルに書き込む
        rect_image_path = path_face_rect + '/' + name + '_rect_' + '.png'
        print("rect_image_path=%s" % rect_image_path)
        cv2.imwrite(rect_image_path, image)
    else:
        print("no face")

    return rect_image_path

def detect_face(image_file):
    # ファイルパス取得
    path = os.path.dirname(image_file)
    name, _ = os.path.splitext( os.path.basename(image_file))

    # カスケードファイルのパスを指定
    cascade_file = "/usr/local/var/pyenv/versions/anaconda3-2.5.0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"

    # 画像の読み込み
    image = cv2.imread(image_file)
    # グレースケールに変換
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔認識用特徴量ファイルを読み込む
    cascade = cv2.CascadeClassifier(cascade_file)
    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(100,100))

    #ディレクトリの作成
    if len(face_list) > 0:
        path_face_rect = path + '/face_rect'
        if os.path.isdir(path_face_rect):
            shutil.rmtree(path_face_rect)
            os.mkdir(path_face_rect)
        else:
            os.mkdir(path_face_rect)

    rect_image_path = ''
    if len(face_list) > 0:
        # 認識した部分を囲む
        i = 0
        for face in face_list:
            x,y,w,h = face
            color = (255, 255, 255) # (B,G,R)
            text = 'none'
            # 枠を作成する
            cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness=2)
            # cv2.putText(画像, 文字, 左下座標, フォント, 文字の大きさ, 色, 文字の太さ, 線の種類)
            cv2.putText(image, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 16)
            i += 1

        # 描画結果をファイルに書き込む
        rect_image_path = path_face_rect + '/' + name + '_rect_' + '.png'
        print("rect_image_path=%s" % rect_image_path)
        cv2.imwrite(rect_image_path, image)
    else:
        print("no face")

    return rect_image_path

def trim_face(image_file):
    # ファイルパス取得
    path = os.path.dirname(image_file)
    name, _ = os.path.splitext( os.path.basename(image_file))

    # カスケードファイルのパスを指定
    cascade_file = "/usr/local/var/pyenv/versions/anaconda3-2.5.0/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"

    # 画像の読み込み
    image = cv2.imread(image_file)
    # グレースケールに変換
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔認識用特徴量ファイルを読み込む
    cascade = cv2.CascadeClassifier(cascade_file)
    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(100,100))

    path_face_trim = ''
    #ディレクトリの作成
    if len(face_list) > 0:
        path_face_trim = path + '/face_trim'
        if os.path.isdir(path_face_trim):
            shutil.rmtree(path_face_trim)
            os.mkdir(path_face_trim)
        else:
            os.mkdir(path_face_trim)

    faces = []
    i = 0
    for face in face_list:
        #顔だけ切り出して保存
        x,y,w,h = face
        trmimage = image[y:y+h, x:x+w]
        trim_image_path = path_face_trim + '/' + name + '_trim_' + str(i) + '.png'
        print("trim_image_path=%s" % trim_image_path)
        cv2.imwrite(trim_image_path, trmimage)
        faces.append(face)
        i += 1

    return path_face_trim, faces

if __name__ == '__main__':
    main()
