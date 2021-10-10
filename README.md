# testOpenCVSharpDnn

![](https://github.com/eiichiromomma/testOpenCVSharpDnn/blob/master/testOpenCVSharpDnn.png)

[OpenCvSharp](https://github.com/shimat/opencvsharp)でとWindows FormsアプリケーションでDnnモジュールを使う例。

* [HttpClientProgressExtensions](https://gist.github.com/dalexsoto/9fd3c5bdbe9f61a717d47c5843384d11)
* [OpenCvSharp Dnn module with Yolo2(OpenCVSharpでのDnnモジュールの使い方)](https://github.com/died/OpenCvSharpDnnYolo)

## Nuget

プロジェクトを作ってNugetパッケージからOpenCvSharp4.WindowsとOpenCvSharp4.runtime.winを入れてる。たぶんOpenCvSharp4とOpenCvSharp4.runtime.winが正しいような気もするが気にしない。

## OpenCvSharpのDnn

ソースの通りなのだが
1. CvDnn.ReadNetFrom* で推論用のファイルを読む
2. BackendとTargetを指定(しなければCPUになる)
3. CvDnn.BlobFromImageで推論用にデータ(blob)を整形
4. SetInputでblobを渡す
5. 出力層を取得してForwardで推論

という流れ。後は最も良い結果を抽出して表示なりする。

## Forms上での画像への書き込みとBBox描画

画像をImage.FromFile→Graphics.FromImageでcanvasを取得して，そこにDrawRectangleとかDrawStringしたうえでPictureBoxの画像に指定する。
貰うのがBBoxの中心(x,y)と幅高さな点に注意。

## 必要なファイルのダウンロード状況のプログレスバー表示

非同期にダウンロードしたかったのでasyncを使うまでは良かったが，プログレスバーのやりかたが分からず探してたら
[HttpClientProgressExtensions](https://gist.github.com/dalexsoto/9fd3c5bdbe9f61a717d47c5843384d11)が全部解決してくれた。

