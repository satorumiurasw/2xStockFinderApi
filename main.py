from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pydantic import field_validator
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import pickle
import json
import requests
from bs4 import BeautifulSoup

# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class statement(BaseModel):
    long_term_loans: Optional[float] = None  # 長期借入金  
    bps: Optional[float] = None  # BPS (1株あたり純資産)  
    investing_cf: Optional[float] = None  # 投資CF
    financing_cf: Optional[float] = None  # 財務CF
    operating_cf_margin: Optional[float] = None  # 営業CFマージン
    revenue: Optional[float] = None  # 売上高  
    # roe: Optional[float] = None  # ROE (自己資本利益率) = 純利益 / 株主資本 * 100
    net_income: Optional[float] = None  # 純利益  
    shareholders_equity: Optional[float] = None  # 株主資本  
    dividend_payout_ratio: Optional[float] = None  # 配当性向
    stock_price: Optional[float] = None  # 株価
    sector: Optional[str] = None  # 業種

    @field_validator(
        "long_term_loans", "bps", "investing_cf", "financing_cf", 
        "operating_cf_margin", "revenue", "net_income", 
        "shareholders_equity", "dividend_payout_ratio", "stock_price",
        mode="before"
    )
    def none_to_nan(cls, v):
        return np.nan if v is None else v
    
# 学習済みのモデルの読み込み
model = pickle.load(open('model6.pkl', 'rb'))
model6_columns = []
with open('model6_columns.json', 'r', encoding='utf-8') as f:
    model6_columns = json.load(f)

# トップページ
@app.get('/')
async def index():
    return {"2xStock": '2xStock_prediction'}

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/make_predictions')
async def make_predictions(features: statement):
    roe = np.nan
    if features.shareholders_equity != np.nan:
        roe = (features.net_income / features.shareholders_equity) * 100
    x = pd.DataFrame([[
        features.long_term_loans,
        features.bps,
        features.investing_cf,
        features.financing_cf,
        features.operating_cf_margin,
        features.revenue,
        roe,
        features.dividend_payout_ratio,
        features.stock_price,
        features.sector,
        ]], columns=['長期借入金', 'BPS', '投資CF', '財務CF', '営業CFマージン', '売上高', 'ROE', '配当性向', '株価', '業種'])
    
    x = pd.get_dummies(x, columns=['業種']).reindex(columns=model6_columns, fill_value=False) # カテゴリカル変数である業種をワンホットエンコーディング
    y = model.predict(x)[0]
    y_proba = model.predict_proba(x)[0]
    return({'predict': float(y), 'proba': float(y_proba[y])})

def get_open_price(soup, year):
    table = soup.find('table', class_='stock_kabuka_dwm')

    # テーブルをDataFrameに変換
    if table:
        # ヘッダーを取得
        headers = [th.text.strip() for th in table.find('thead').find_all('th')]

        # データ行を取得
        rows = []
        for row in table.find('tbody').find_all('tr'):
            cells = []
            for cell in row.find_all(['th', 'td']):
                # <time>タグや<span>タグの内部テキストを含む処理
                if cell.find('time'):
                    cells.append(cell.find('time').text.strip())
                elif cell.find('span'):
                    cells.append(cell.find('span').text.strip())
                else:
                    cells.append(cell.text.strip())
            rows.append(cells)

        # DataFrameの作成
        df = pd.DataFrame(rows, columns=headers)

        return df.at[df[df['日付'] == f'{year - 2000}/01/01'].index[0], '始値']
    
def get_company_name_and_price(code, year):
    response = requests.get(f'https://kabutan.jp/stock/kabuka?code={code}&ashi=yar')
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        div = soup.find('div', class_='si_i1_1')
        company_name = div.find('h2').text.replace(f'{code}　', '').strip()
        current_price = soup.find('span', class_='kabuka').text.replace('円', '').strip()
        open_price = get_open_price(soup, year)
    return company_name, float(current_price.replace(',', '')), open_price
