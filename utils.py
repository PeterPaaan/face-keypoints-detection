import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import cv2


def load_data(test=False):
    """
    当 test 为真, 加载测试数据, 否则加载训练数据 
    """
    FTRAIN = './data/training.csv'
    FTEST = './data/test.csv'
    fname = FTEST if test else FTRAIN
    df = read_csv(fname)

    # 将'Image' 列中 '空白键' 分割的数字们转换为一个 numpy array
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # 丢弃有缺失值的数据
    df = df.dropna()  

    # 将图像的数字从 0 到 255 的整数转换为 0 到 1 的实数
    X = np.vstack(df['Image'].values) / 255.  
    X = X.astype(np.float32)

    # 将 X 的每一行转换为一个 96 * 96 * 1 的三维数组
    X = X.reshape(-1, 96, 96, 1) 

    # 只有 FTRAIN 包含关键点的数据 (target value)
    if not test:  
        y = df[df.columns[:-1]].values
        # 将关键点的值 normalize 到 [-1, 1] 之间
        y = (y - 48) / 48  
        # 置乱训练数据
        X, y = shuffle(X, y, random_state=42)  
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def transparentOverlay(src , overlay , pos=(0,0), scale = 1):
    """
    将带透明通道(png图像)的图像 overlay 叠放在src图像上方
    :param src: 背景图像
    :param overlay: 带透明通道的图像 (BGRA)
    :param pos: 叠放的起始位置
    :param scale : overlay图像的缩放因子
    :return: Resultant Image
    """
    if scale != 1:
        overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)

    # overlay图像的高和宽
    h,w,_ = overlay.shape  
    # 叠放的起始坐标
    y,x = pos[0],pos[1] 
    
    # 以下被注释的代码是没有优化的版本, 便于理解, 与如下没有注释的版本的功能一样
    """     
    # src图像的高和宽
    rows,cols,_ = src.shape  
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # 读取alpha通道的值
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src """

    alpha = overlay[:,:,3]/255.0
    alpha = alpha[..., np.newaxis]
    src[x:x+h,y:y+w,:] = alpha * overlay[:,:,:3] + (1-alpha)*src[x:x+h,y:y+w,:]
    return src