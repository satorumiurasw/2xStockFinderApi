from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier
import pickle

# インスタンス化
app = FastAPI()

# 入力するデータ型の定義
class doubleStock(BaseModel):
    bps: float  # BPS (1株あたり純資産)  
    long_term_loans: float  # 長期借入金  
    dividend_payout_ratio: float  # 配当性向  
    revenue: float  # 売上高  
    roe: float  # ROE (自己資本利益率)  
    investing_cf: float  # 投資CF
    operating_cf_margin: float  # 営業CFマージン  
    financing_cf: float  # 財務CF 
    shareholders_equity: float  # 株主資本  
    retained_earnings: float  # 利益剰余金  
    cash_equivalents: float  # 現金同等物  
    roa: float  # ROA (総資産利益率)  
    dividend_per_share: float  # 一株配当  
    retained_earnings_dividends: float  # 剰余金の配当  
    net_assets: float  # 純資産  
    total_assets: float  # 総資産  
    net_income: float  # 純利益  
    ordinary_income: float  # 経常利益  
    net_asset_dividend_ratio: float  # 純資産配当率 
    equity_ratio: float  # 自己資本比率  
    operating_income: float  # 営業利益  
    total_return_ratio: float  # 総還元性向  
    short_term_loans: float  # 短期借入金  
    capital_expenditure: float  # 設備投資  
    eps: float  # EPS (1株当たり利益)  
    share_buyback: float  # 自社株買い  
    operating_cf: float  # 営業CF  

# 学習済みのモデルの読み込み
model = pickle.load(open('model4_simple', 'rb'))

# トップページ
@app.get('/')
async def index():
    return {"2xStock": '2xStock_prediction'}

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/make_predictions')
async def make_predictions(features: doubleStock):
    return({'prediction':str(model.predict([[
        features.long_term_loans,
        features.bps,
        features.investing_cf,
        features.financing_cf,
        features.operating_cf_margin,
        features.revenue,
        features.roe,
        features.dividend_payout_ratio
    ]])[0])})
