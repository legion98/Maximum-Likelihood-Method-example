from PIL import Image
import numpy as np

img = Image.open('irabu_zhang1.bmp')  #画像読み込み
rgb_im = img.convert('RGB')
width, height = img.size
output = Image.new('RGB', (width,height)) #出力用ファイル
img_pixels = np.array([[rgb_im.getpixel((x,y)) for x in range(width)] for y in range(height)])
test_data_coodinate = np.array([[[70, 188], [52, 193], [263, 58], [40, 255], [280, 118], [241, 6], [280, 98], [269, 42], [166, 196], [222, 145] ],
        [ [187, 288], [238, 304], [241, 264], [229, 314], [259, 283], [246, 285], [270, 270], [229, 252], [217, 299], [239, 298] ],
        [ [327, 270], [325, 254], [338, 248], [328, 298], [357, 279], [313, 245], [200, 278], [317, 234], [358, 295], [364, 302] ],
        [ [256, 320], [357, 21], [344, 118], [347, 241], [369, 271], [307, 216], [350, 23], [374, 117], [384, 260], [344, 115] ],
        [ [322, 134], [293, 182], [283, 174], [316, 130], [315, 155], [290, 172], [320, 139], [301, 174], [309, 162], [294, 157] ]])  #教師用データ（座標）
test_data = np.zeros([5,10,3])  #教師用データ（r,g,b）
mean_vector = np.zeros([5,3])  #平均ベクトル
covariance_matrix = np.zeros([3,3])  #各クラス共分散行列計算用
covariance_matrix_arrange = np.zeros([5,3,3])  #各クラス共分散行列
inverse_matrix = np.zeros([3,3]) #各クラス共分散行列の逆行列計算用
inverse_matrix_arrange = np.zeros([5,3,3])  #各クラス共分散行列の逆行列
mahalanobis = np.zeros(5)  #マハラノビス距離算出用
data_colors = np.zeros([height,width]) #各画素がどのクラスに入るかを格納

for i in range(5): #教師データを格納
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

for i in range(height): #最尤推定法の計算
    for j in range(width):
        temp=10000
        for k in range(5):
            mahalanobis[k] = np.log(np.linalg.det(covariance_matrix_arrange[k]))+np.dot(np.dot((img_pixels[i][j] - mean_vector[k]).T,inverse_matrix_arrange[k]),img_pixels[i][j] - mean_vector[k])
            if temp>mahalanobis[k]:
                temp=mahalanobis[k]
                data_colors[i][j]=k+1
                output.putpixel((j,i), (int(mean_vector[k][0]), int(mean_vector[k][1]), int(mean_vector[k][2])))  #各クラスに対応する平均色を参照

output.save('result.bmp')
