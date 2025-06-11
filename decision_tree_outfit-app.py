import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 載入資料（建議你將下列資料存為 outfit_formal_dataset.csv）
data = {
    "性別": ["女", "女", "男", "男", "女", "女", "女", "女", "男", "男", "男", "女", "女", "男", "男"],
    "天氣": ["晴天", "雨天", "晴天", "雨天", "晴天", "晴天", "晴天", "雨天", "晴天", "晴天", "雨天", "晴天", "雨天", "晴天", "雨天"],
    "場合": ["上學", "上學", "上學", "上學", "工作", "工作", "工作", "工作", "工作", "工作", "工作", "約會", "約會", "約會", "約會"],
    "推薦上衣": ["亮色條紋短袖上衣", "長款風衣", "牛仔夾克", "防水連帽夾克", "白色法國袖罩衫", "針織POLO衫", "短版外套", "可收納風衣", "深色西裝外套", "休閒西裝外套＋素色T恤", "防潑水翻領大衣", "開領襯衫", "寬鬆襯衫裙", "藍色格紋襯衫", "挺括西裝外套"],
    "推薦下裝": ["寬版休閒長褲（卡其綠）", "短裙（內搭）", "牛仔褲", "百慕大短褲", "抽褶西裝褲", "白色長裙", "寬腿羊毛長褲", "通勤長褲", "配套西裝長褲", "合身休閒長褲", "西裝長褲", "黑色無袖連衣裙", "N/A", "淺卡其色長褲", "九分西褲"]
}

df = pd.DataFrame(data)
print("原始資料：")
print(df.head())

# 編碼類別文字資料為數字
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

print("\n編碼後的資料：")
print(df.head())

# 分割特徵與標籤
X = df[['性別', '天氣', '場合']]
y_top = df['推薦上衣']
y_bottom = df['推薦下裝']

# 建立並訓練模型
tree_top = DecisionTreeClassifier()
tree_bottom = DecisionTreeClassifier()
tree_top.fit(X, y_top)
tree_bottom.fit(X, y_bottom)

# 模擬使用者輸入（例如：女、晴天、工作）
user_input = pd.DataFrame([[le.transform(['女'])[0],
                            le.transform(['晴天'])[0],
                            le.transform(['工作'])[0]]],
                          columns=['性別', '天氣', '場合'])

# 預測
top_pred = tree_top.predict(user_input)
bottom_pred = tree_bottom.predict(user_input)

# 還原文字標籤
top_label = le.inverse_transform([top_pred[0]])[0]
bottom_label = le.inverse_transform([bottom_pred[0]])[0]

print(f"推薦穿搭：{top_label} + {bottom_label}")
