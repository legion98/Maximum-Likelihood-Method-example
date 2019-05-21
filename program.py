from PIL import Image
import numpy as np

#画像の特定の５箇所を５クラスと設定して，該当するクラスごとに10点のr,g,b値を教師データとして格納し，
#入力画像の各画素がどのクラスに該当するかを最尤法を用いて算出する

img = Image.open('sample.bmp')  #画像読み込み
rgb_im = img.convert('RGB')
width, height = img.size
output = Image.new('RGB', (width,height)) #出力用ファイル
img_pixels = np.array([[rgb_im.getpixel((x,y)) for x in range(width)] for y in range(height)])
#以下任意に各クラス内の座標を指定 (width,heightに収まる値で)
test_data_coodinate = np.array([[[1, 1], [10, 1], [20, 1], [30, 1], [40, 1], [1, 10], [1, 20], [1, 30], [1, 40], [1, 50] ],
        [ [1, 2], [10, 2], [20, 2], [30, 2], [40, 2], [2, 10], [2, 20], [2, 30], [2, 40], [2, 50] ],
        [ [1, 3], [10, 3], [20, 3], [30, 3], [40, 3], [3, 10], [3, 20], [3, 30], [3, 40], [3, 50] ],
        [ [1, 4], [10, 4], [20, 4], [30, 4], [40, 4], [4, 10], [4, 20], [4, 30], [4, 40], [4, 50] ],
        [ [1, 5], [10, 5], [20, 5], [30, 5], [40, 5], [5, 10], [5, 20], [5, 30], [5, 40], [5, 50] ]])  #教師用データ（座標）
test_data = np.zeros([5,10,3])  #教師用データ（r,g,b）
mean_vector = np.zeros([5,3])  #平均ベクトル
covariance_matrix = np.zeros([3,3])  #各クラス共分散行列計算用
covariance_matrix_arrange = np.zeros([5,3,3])  #各クラス共分散行列
inverse_matrix = np.zeros([3,3]) #各クラス共分散行列の逆行列計算用
inverse_matrix_arrange = np.zeros([5,3,3])  #各クラス共分散行列の逆行列
mahalanobis = np.zeros(5)  #マハラノビス距離算出用
data_colors = np.zeros([height,width]) #各画素がどのクラスに入るかを格納

for i in range(5): #教師データ(r,g,b)を格納
    for j in range(10):
        test_data[i][j]= rgb_im.getpixel((int(test_data_coodinate[i][j][0]),int(test_data_coodinate[i][j][1])))

for i in range (5): #平均ベクトルを算出
    mean_vector[i]=  np.mean(test_data[i],axis=0)
i=0
for data in test_data: #共分散行列と共分散行列の逆行列を計算
    covariance_matrix = np.cov(data,rowvar=False)
    inverse_matrix = np.linalg.inv(covariance_matrix)
    for j in range(3):
        for k in range(3):
            covariance_matrix_arrange[i][j][k]=covariance_matrix[j][k]
            inverse_matrix_arrange[i][j][k] = inverse_matrix[j][k]
    i+=1

for i in range(height): #最尤法の計算
    for j in range(width):
        temp=10000
        for k in range(5):
            mahalanobis[k] = np.log(np.linalg.det(covariance_matrix_arrange[k]))+np.dot(np.dot((img_pixels[i][j] - mean_vector[k]).T,inverse_matrix_arrange[k]),img_pixels[i][j] - mean_vector[k])
            if temp>mahalanobis[k]:
                temp=mahalanobis[k]
                data_colors[i][j]=k+1
                output.putpixel((j,i), (int(mean_vector[k][0]), int(mean_vector[k][1]), int(mean_vector[k][2])))  #各クラスに対応する平均色を参照

output.save('result.bmp')
