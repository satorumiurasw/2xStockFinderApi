from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from xgboost import XGBClassifier
import pickle

# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class statement(BaseModel):
    bps: float  # BPS (1株あたり純資産)  
    long_term_loans: float  # 長期借入金  
    dividend_payout_ratio: float  # 配当性向  
    revenue: float  # 売上高  
    roe: float  # ROE (自己資本利益率)  
    investing_cf: float  # 投資CF
    operating_cf_margin: float  # 営業CFマージン  
    financing_cf: float  # 財務CF 
    shareholders_equity: Optional[float] = None  # 株主資本  
    retained_earnings: Optional[float] = None  # 利益剰余金  
    cash_equivalents: Optional[float] = None  # 現金同等物  
    roa: Optional[float] = None  # ROA (総資産利益率)  
    dividend_per_share: Optional[float] = None  # 一株配当  
    retained_earnings_dividends: Optional[float] = None  # 剰余金の配当  
    net_assets: Optional[float] = None  # 純資産  
    total_assets: Optional[float] = None  # 総資産  
    net_income: Optional[float] = None  # 純利益  
    ordinary_income: Optional[float] = None  # 経常利益  
    net_asset_dividend_ratio: Optional[float] = None  # 純資産配当率 
    equity_ratio: Optional[float] = None  # 自己資本比率  
    operating_income: Optional[float] = None  # 営業利益  
    total_return_ratio: Optional[float] = None  # 総還元性向  
    short_term_loans: Optional[float] = None  # 短期借入金  
    capital_expenditure: Optional[float] = None  # 設備投資  
    eps: Optional[float] = None  # EPS (1株当たり利益)  
    share_buyback: Optional[float] = None  # 自社株買い  
    operating_cf: Optional[float] = None  # 営業CF  

# 学習済みのモデルの読み込み
model = pickle.load(open('model4_simple', 'rb'))

# トップページ
@app.get('/')
async def index():
    return {"2xStock": '2xStock_prediction'}

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/make_predictions')
async def make_predictions(features: statement):
    x = [
        features.long_term_loans,
        features.bps,
        features.investing_cf,
        features.financing_cf,
        features.operating_cf_margin,
        features.revenue,
        features.roe,
        features.dividend_payout_ratio,
    ]
    prediction = model.predict([x])[0]
    return({'prediction': str(prediction)})
