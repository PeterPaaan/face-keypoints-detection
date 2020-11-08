from utils import load_data
import kmodel  # _ref as kmodel

# 加载训练数据
X_train, y_train = load_data()

# 创建网络结构
my_model = kmodel.create_model()

# 编译网络模型
kmodel.compile_model(my_model)

# 训练网络模型
kmodel.train_model(my_model, X_train, y_train)

# 保存网络模型
kmodel.save_model(my_model, 'my_model')

# end


